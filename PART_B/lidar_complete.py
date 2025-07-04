import sys
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import cv2
from pathlib import Path
import argparse
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

LOCAL_HEIGHT_TRESHOLD = 0.1 # adaptive_height_filtering
HEIGHT_VARIATION_THRESHOLD = 0.1  # road_continuity_filter
#--image_dir=C:\Users\USER\Documents\GitHub\Camera_vs_LIDAR\PART_B\fakeleft --velodyne_dir=C:\Users\USER\Documents\GitHub\Camera_vs_LIDAR\PART_B\fakebin --calib_dir=C:\Users\USER\Documents\GitHub\Camera_vs_LIDAR\PART_B\fakecalib

# ----------------- basic helpers ----------------- # 
def get_args():
    p = argparse.ArgumentParser("KITTI Velodyne viewer + road RANSAC")
    p.add_argument("--velodyne_dir", default="C:/Users/USER/Documents/_CAMERA_LIDAR/data_road_velodyne/training/velodyne")
    #p.add_argument("--velodyne_dir", default="C:/Users/Mania/Documents/KITTI/data_road_velodyne/training/velodyne")
    p.add_argument("--index", default="all")
    #p.add_argument("--dist", type=float, default=0.15) not used functionality
    #p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--calib_dir", default="C:/Users/USER/Documents/_CAMERA_LIDAR/data_road/training/calib")
    p.add_argument("--image_dir",  default="C:/Users/USER/Documents/_CAMERA_LIDAR/data_road/training/image_2")
    p.add_argument("--curbs", action="store_true", help="Enable curb visualization")
    p.add_argument("--testing", action="store_true", help="Use KITTI testing set instead of training")
    p.add_argument("--kitti", action="store_true", help="Run full KITTI training pipeline")
    #p.add_argument("--calib_dir", default="C:/Users/Mania/Documents/KITTI/data_road/training/calib")
    #p.add_argument("--image_dir",  default="C:/Users/Mania/Documents/KITTI/data_road/training/image_2")
    p.add_argument("--video", action="store_true", help="Enable automatic video playback mode")
    return p.parse_args()


# KITTI dataset processing
def load_bin(bin_path):
    """Load binary point cloud file"""
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]

def parse_calib(txt: Path):
    """Parse calibration file"""
    data = {}
    with open(txt) as f:
        for line in f:
            if ':' in line:
                k, v = line.strip().split(':', 1)
                data[k] = np.fromstring(v, sep=' ')
    P2 = data['P2'].reshape(3, 4)
    R0 = data['R0_rect'].reshape(3, 3)
    Tr = data['Tr_velo_to_cam'].reshape(3, 4)
    T = np.eye(4); T[:3, :4] = Tr
    R = np.eye(4); R[:3, :3] = R0
    P = np.eye(4); P[:3, :4] = P2
    return P @ R @ T

def project(pts, P):
    """Project 3D points to image coordinates"""
    h = np.hstack([pts, np.ones((len(pts), 1))])
    uvw = (P @ h.T).T
    z = uvw[:, 2]
    valid = z > 0
    uv = np.zeros((len(pts), 2), dtype=int)
    uv[valid] = (uvw[valid, :2] / z[valid, np.newaxis]).astype(int)
    return uv[valid]

def pc_to_o3d(xyz):
    '''Convert numpy array to Open3D PointCloud'''
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    return pc

def project_all_points(pts, P, img_shape):
    '''Project 3D points to image coordinates, return only valid points'''
    if len(pts) == 0:
        return np.array([]).reshape(0, 2)
    
    # Convert to homogeneous coordinates
    h = np.hstack([pts, np.ones((len(pts), 1))])
    
    # Project
    uvw = (P @ h.T).T
    z = uvw[:, 2]
    
    # Keep points in front of camera
    valid = z > 0.1  # Small threshold to avoid division by very small numbers
    
    if not np.any(valid):
        return np.array([]).reshape(0, 2)
    
    # Convert to image coordinates
    uv = uvw[valid, :2] / z[valid, np.newaxis]
    uv = uv.astype(int)
    
    # Filter points within image bounds
    img_valid = ((uv[:, 0] >= 0) & (uv[:, 0] < img_shape[1]) & 
                (uv[:, 1] >= 0) & (uv[:, 1] < img_shape[0]))
    
    return uv[img_valid]

####################################################################################################
# —————————————————————————————————————— B1 road detetcion  —————————————————————————————————————— #  
####################################################################################################

def multi_plane_ransac(pcd, max_planes=3, dist_thresh=0.1, min_points=100):
    '''Find multiple ground planes using iterative RANSAC'''
    points = np.asarray(pcd.points)
    remaining_indices = set(range(len(points))) 
    # all integer indices of points
    planes = []
    
    for _ in range(max_planes):
        if len(remaining_indices) < min_points:
            break
            
        # Create sub-cloud from remaining points
        sub_indices = list(remaining_indices) # initially all points
        sub_pcd = o3d.geometry.PointCloud()
        sub_pcd.points = o3d.utility.Vector3dVector(points[sub_indices])
        
        # RANSAC on sub-cloud
        plane, inliers = sub_pcd.segment_plane( 
            distance_threshold=dist_thresh, 
            ransac_n=3, 
            num_iterations=1000
        )  # plane: ax + by + cz + d = 0 
        # inliers: list of integers: Each is the index of a point in the input point cloud (pcd.points) that lies close enough to the plane, i.e., within distance_threshold.
        
        if len(inliers) < min_points:
            break
        
        # Convert back to original indices
        original_inliers = [sub_indices[i] for i in inliers] # because inliers are indices of sub_pcd, we need to map them back to the original point cloud
        planes.append((plane, original_inliers))
        
        # Remove found inliers from remaining points
        remaining_indices -= set(original_inliers)
    
    return planes

def adaptive_height_filtering(points, grid_size=1.0, percentile=10):
    '''Adaptive height filtering based on local ground level
    Makes use of a 2D grid to estimate local ground height and filters points that are too high above it.'''
    if len(points) == 0:
        return np.array([]).reshape(0, 3)
    
    # Create 2D grid
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    x_bins = int((x_max - x_min) / grid_size) + 1
    y_bins = int((y_max - y_min) / grid_size) + 1
    
    # Digitize points into grid
    x_idx = ((points[:, 0] - x_min) / grid_size).astype(int)
    y_idx = ((points[:, 1] - y_min) / grid_size).astype(int)
    
    # Ensure indices are within bounds
    x_idx = np.clip(x_idx, 0, x_bins - 1)
    y_idx = np.clip(y_idx, 0, y_bins - 1)
    
    # Calculate local ground level for each grid cell
    ground_levels = np.full((x_bins, y_bins), np.nan)
    
    for i in range(x_bins):
        for j in range(y_bins):
            mask = (x_idx == i) & (y_idx == j)
            if np.sum(mask) > 0:
                ground_levels[i, j] = np.percentile(points[mask, 2], percentile)
    
    # Interpolate missing values
    from scipy.interpolate import griddata
    valid_mask = ~np.isnan(ground_levels)
    if np.sum(valid_mask) > 3:  # Need at least 3 points for interpolation
        xi, yi = np.meshgrid(range(x_bins), range(y_bins), indexing='ij')
        valid_points = np.column_stack([xi[valid_mask], yi[valid_mask]])
        valid_values = ground_levels[valid_mask]
        
        all_points = np.column_stack([xi.ravel(), yi.ravel()])
        interpolated = griddata(valid_points, valid_values, all_points, method='linear')
        ground_levels = interpolated.reshape(x_bins, y_bins)
    
    # Filter points based on local ground level
    filtered_indices = []
    for idx, (x, y, z) in enumerate(points):
        local_ground = ground_levels[x_idx[idx], y_idx[idx]]
        if not np.isnan(local_ground) and (z - local_ground) < LOCAL_HEIGHT_TRESHOLD:  # 10cm above local ground
            filtered_indices.append(idx)
    
    return points[filtered_indices]

def cluster_road_segments(points, eps=1.0, min_samples=10):
    '''Cluster road points into coherent segments'''
    if len(points) < min_samples:
        return [points]
    
    # Use DBSCAN clustering on 2D coordinates
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(points[:, :2])
    
    # Group points by cluster
    clusters = []
    for label in set(labels):
        if label != -1:  # Ignore noise points
            cluster_points = points[labels == label]
            if len(cluster_points) >= min_samples:
                clusters.append(cluster_points)
    
    return clusters

#unused
def road_continuity_filter(points, search_radius=2.0, min_neighbors=5):
    '''Filter points based on road continuity
    Keep only the points that:
        1. Have enough nearby neighbors (within a radius)
        2. Whose neighbors lie roughly on the same horizontal surface (low Z variation)'''
    
    if len(points) < min_neighbors:
        return points
    
    tree = cKDTree(points[:, :2])  # 2D spatial tree #Fast lookup structure to find neighbors in the horizontal plane
    valid_indices = []
    
    for i, point in enumerate(points):
        # Find neighbors in 2D
        neighbors = tree.query_ball_point(point[:2], search_radius) #For each point, find its 2D neighbors within search_radius
        
        if len(neighbors) >= min_neighbors:
            neighbor_points = points[neighbors]
            
            # Check height consistency
            height_std = np.std(neighbor_points[:, 2]) # standard deviation of Z values of neighbors
            if height_std < HEIGHT_VARIATION_THRESHOLD:  # Low height variation indicates road
                valid_indices.append(i)
    
    return points[valid_indices]

# —————————————————————————————————————— sidewalk —————————————————————————————————————— #

# NORMALS - PCA - SIDEWALK REMOVAL
def compute_normals(points, radius=0.3):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    return np.asarray(pcd.normals)

def pca_high_variation_mask(points, radius=0.3, z_variance_thresh=0.002):
    """
    Return a boolean mask of points with high vertical roughness in local patch.
    Useful for excluding curb-like areas from road points.
    """
    kdt = cKDTree(points[:, :2])
    high_variance_mask = np.zeros(len(points), dtype=bool)
    
    for i, p in enumerate(points):
        idx = kdt.query_ball_point(p[:2], radius) 
        if len(idx) < 5:
            continue
        patch = points[idx]
        pca = PCA(n_components=3).fit(patch)
        # The 3rd eigenvalue (variance) is largest if there's a jump/step
        if pca.explained_variance_[2] > z_variance_thresh:
            high_variance_mask[i] = True
    return high_variance_mask

def remove_sidewalks(road_points):
    # Cluster road segments
    road_clusters = cluster_road_segments(road_points, eps=1.5, min_samples=15)
    
    # Select the largest cluster as the main road
    if road_clusters:
        main_road = max(road_clusters, key=len)
        main_road = main_road[np.abs(main_road[:, 1]) < 5.0]

        pca_curb_mask = pca_high_variation_mask(main_road, radius=0.3, z_variance_thresh=0.0001)
        rough_points = main_road[pca_curb_mask]      # yellow
        main_road = main_road[~pca_curb_mask]     # blue
    else:
        main_road = np.array([]).reshape(0, 3)
        rough_points = np.array([]).reshape(0, 3)
    # Median y of road
    road_center_y = np.median(main_road[:, 1]) if len(main_road) > 0 else 0
    #print(main_road.shape)
    #road_center_y = 0.0
    left_curb_y, right_curb_y,left_rough,right_rough = split_rough_by_side(rough_points, road_center_y)    
    main_road = main_road[(main_road[:, 1] > left_curb_y) & (main_road[:, 1] < right_curb_y)]   
 
    return main_road,left_rough, right_rough

def split_rough_by_side(rough_points, center_y, MIN_CANDIDATES=300):
    """
    Splits rough curb points into left/right and computes curb boundaries.
    
    Returns:
    - left_curb_y: rightmost Y value of left curb
    - right_curb_y: leftmost Y value of right curb
    - left_rough, right_rough: the split point clouds
    """
    left_rough = rough_points[rough_points[:, 1] < center_y]
    right_rough = rough_points[rough_points[:, 1] > center_y]

    left_curb_y = np.percentile(left_rough[:, 1], 95) if len(left_rough) > MIN_CANDIDATES else -np.inf   # rightmost Y of left curb
    right_curb_y = np.percentile(right_rough[:, 1], 5) if len(right_rough) > MIN_CANDIDATES else np.inf  # leftmost Y of right curb

    return left_curb_y, right_curb_y, left_rough, right_rough

# def filter_connected_points(points, search_radius=1.0, min_neighbors=5):
#     """
#     Given a set of points (e.g., rough/curb points), keep only those that
#     have enough nearby neighbors — i.e., are not isolated.
#     """
#     if len(points) < min_neighbors:
#         return np.empty((0, 3))

#     tree = cKDTree(points[:, :2])  # 2D neighborhood
#     keep = []

#     for i, pt in enumerate(points):
#         neighbors = tree.query_ball_point(pt[:2], search_radius)
#         if len(neighbors) >= min_neighbors:
#             keep.append(i)

#     return points[keep]

#UNUSED:
def find_curb_by_normals(points, verticality_thresh=0.1):
    '''normal close to 0 means the point is on a curb(vertical surface)'''
    normals = compute_normals(points)
    verticality = np.abs(normals[:, 2])
    curb_mask = verticality < verticality_thresh
    return curb_mask


def get_largest_cluster(points, eps=0.5, min_samples=10):
    if len(points) == 0:
        return np.empty((0, 3))

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_

    # -1 means noise
    valid_mask = labels != -1
    labels = labels[valid_mask]
    points = points[valid_mask]

    if len(labels) == 0:
        return np.empty((0, 3))

    # Get label with most points
    biggest_label = np.bincount(labels).argmax()
    return points[labels == biggest_label]


def detect_curb_by_height_discontinuity(points, radius=0.25, z_jump_thresh=0.12): 
    """
    Returns a boolean mask of points that are near vertical discontinuities in z.
    """
    from scipy.spatial import cKDTree
    mask = np.zeros(len(points), dtype=bool)
    kdt = cKDTree(points[:, :2])
    for i, p in enumerate(points):
        idx = kdt.query_ball_point(p[:2], radius)
        if len(idx) < 3:
            continue
        local_z = points[idx, 2]
        if local_z.max() - local_z.min() > z_jump_thresh:
            mask[i] = True
    return mask

def filter_clusters_by_size(points, eps=0.8, min_samples=10, min_cluster_size=350):
    if len(points) < min_samples:
        return []
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, :2])
    labels = db.labels_
    clusters = []
    for lbl in set(labels):
        if lbl == -1:
            continue  # skip noise
        cluster = points[labels == lbl]
        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)
    return clusters

def count_clusters(points, eps=0.8, min_samples=8):
    if len(points) < min_samples:
        return 0
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points[:, :2])
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return num_clusters

#######################################################################################################
# —————————————————————————————————————— B2 OBSTACLE DETECTION —————————————————————————————————————— #  
#######################################################################################################

def detect_obstacles(all_points, road_points, height_threshold=0.5, min_cluster_size=5):
    '''Detect obstacles above the road surface'''
    if len(road_points) == 0:
        return np.array([]).reshape(0, 3)
    
    # Create 2D grid of road heights
    road_tree = cKDTree(road_points[:, :2])
    
    obstacle_points = []
    for point in all_points:
        # Find nearest road points
        distances, indices = road_tree.query(point[:2], k=min(5, len(road_points)))
        
        if len(indices) > 0:
            # Estimate local road height
            local_road_height = np.mean(road_points[indices, 2])
            
            # Check if point is significantly above road
            if point[2] - local_road_height > height_threshold:
                obstacle_points.append(point)
    
    return np.array(obstacle_points)

def cluster_obstacles(obstacle_points, eps=0.8, min_samples=5):
    '''Cluster obstacle points into individual objects'''
    if len(obstacle_points) < min_samples:
        return []
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(obstacle_points)
    
    clusters = []
    for label in set(labels):
        if label != -1:  # Ignore noise points
            cluster_points = obstacle_points[labels == label]
            if len(cluster_points) >= min_samples:
                clusters.append(cluster_points)
    
    return clusters

def get_distance_color(distance):
    '''Get color based on distance (closer = more red, farther = more green)'''
    if distance < 5:
        return (0, 0, 255)  # Red - very close
    elif distance < 10:
        return (0, 100, 255)  # Orange-red
    elif distance < 15:
        return (0, 165, 255)  # Orange
    elif distance < 20:
        return (0, 255, 255)  # Yellow
    elif distance < 30:
        return (100, 255, 100)  # Light green
    else:
        return (0, 255, 0)  # Green - far

def draw_3d_bounding_box(img, points_3d, proj, color=(0, 255, 0), thickness=2):
    '''Draw 3D bounding box on image'''
    if len(points_3d) == 0:
        return
    
    # Calculate bounding box in 3D
    min_vals = np.min(points_3d, axis=0)
    max_vals = np.max(points_3d, axis=0)
    
    # Define 8 corners of the bounding box
    corners_3d = np.array([
        [min_vals[0], min_vals[1], min_vals[2]],  # 0
        [max_vals[0], min_vals[1], min_vals[2]],  # 1
        [max_vals[0], max_vals[1], min_vals[2]],  # 2
        [min_vals[0], max_vals[1], min_vals[2]],  # 3
        [min_vals[0], min_vals[1], max_vals[2]],  # 4
        [max_vals[0], min_vals[1], max_vals[2]],  # 5
        [max_vals[0], max_vals[1], max_vals[2]],  # 6
        [min_vals[0], max_vals[1], max_vals[2]]   # 7
    ])
    
    # Project 3D corners to 2D
    corners_2d = project_all_points(corners_3d, proj, img.shape)
    
    if len(corners_2d) < 8:
        return  # Not enough visible corners
    
    # Draw bottom face (z = min)
    bottom = [0, 1, 2, 3, 0]
    for i in range(len(bottom) - 1):
        pt1 = tuple(corners_2d[bottom[i]])
        pt2 = tuple(corners_2d[bottom[i + 1]])
        cv2.line(img, pt1, pt2, color, thickness)
    
    # Draw top face (z = max)
    top = [4, 5, 6, 7, 4]
    for i in range(len(top) - 1):
        pt1 = tuple(corners_2d[top[i]])
        pt2 = tuple(corners_2d[top[i + 1]])
        cv2.line(img, pt1, pt2, color, thickness)
    
    # Draw vertical edges
    for i in range(4):
        pt1 = tuple(corners_2d[i])
        pt2 = tuple(corners_2d[i + 4])
        cv2.line(img, pt1, pt2, color, thickness)

def draw_distance_text(img, center_2d, distance, color):
    '''Draw distance text near the bounding box'''
    text = f"{distance:.1f}m"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Position text above the center
    text_x = max(0, center_2d[0] - text_width // 2)
    text_y = max(text_height, center_2d[1] - 10)
    
    # Draw background rectangle for better visibility
    cv2.rectangle(img, (text_x - 2, text_y - text_height - 2), 
                  (text_x + text_width + 2, text_y + 2), (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)

def draw_legend(img):
    '''Draw distance color legend on the image'''
    legend_y = 30
    legend_x = img.shape[1] - 200
    
    distances = [5, 10, 15, 20, 30, 40]
    labels = ["<5m", "5-10m", "10-15m", "15-20m", "20-30m", ">30m"]
    
    # Draw legend background
    cv2.rectangle(img, (legend_x - 10, 10), (img.shape[1] - 10, legend_y + len(distances) * 25), 
                  (0, 0, 0), -1)
    cv2.rectangle(img, (legend_x - 10, 10), (img.shape[1] - 10, legend_y + len(distances) * 25), 
                  (255, 255, 255), 2)
    
    # Draw legend title
    cv2.putText(img, "Distance Legend", (legend_x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    for i, (dist, label) in enumerate(zip(distances, labels)):
        color = get_distance_color(dist)
        y_pos = legend_y + 20 + i * 20
        
        # Draw color circle
        cv2.circle(img, (legend_x, y_pos), 5, color, -1)
        
        # Draw label
        cv2.putText(img, label, (legend_x + 15, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# —————————————————————————————————————— B3 —————————————————————————————————————— #  

#BEST
def direction_arrow_road_surface(img, calib_path, obstacle_clusters=None, color=(0, 0, 0), 
                                max_length=6.0, min_length=1.0, safe_distance=15.0):
    """
    Draw arrow on the road surface.
    Arrow length is inversely proportional to obstacle proximity in the forward direction.
    
    Parameters:
    - obstacle_clusters: list of obstacle point clouds
    - max_length: maximum arrow length when no obstacles (meters)
    - min_length: minimum arrow length when obstacles are very close (meters)
    - safe_distance: distance beyond which obstacles don't affect arrow length (meters)
    """

    # Parse calibration
    data = {}
    with open(calib_path) as f:
        for line in f:
            if ':' in line:
                k, v = line.strip().split(':', 1)
                data[k] = np.fromstring(v, sep=' ')

    Tr = np.eye(4)
    Tr[:3, :4] = data['Tr_velo_to_cam'].reshape(3, 4)

    R0 = np.eye(4)
    R0[:3, :3] = data['R0_rect'].reshape(3, 3)

    P2 = data['P2'].reshape(3, 4)
    proj = P2 @ R0 @ Tr  # 3D→2D projection matrix

    # Calculate arrow length based on obstacles
    arrow_length = calculate_adaptive_arrow_length(
        obstacle_clusters, max_length, min_length, safe_distance
    )
    
    # Choose arrow color based on length (optional visual feedback)
    if arrow_length < max_length * 0.4:
        arrow_color = (0, 0, 255)  # Red - very close obstacles
    elif arrow_length < max_length * 0.6:
        arrow_color = (0, 165, 255)  # Orange - moderate obstacles
    elif arrow_length < max_length:
        arrow_color = (255, 0, 255) # magenta - farther obstacles
    else:
        arrow_color = color  # Default color - safe
    
    # Place arrow on road surface (z ≈ -1.7m is typical road height in KITTI)
    road_height = -1.7
    
    # Try different forward distances
    # 5 too small
    for forward_dist in [8, 10, 12, 15, 20]: # WORKS WITH 8 if sth wrong tries other distance
        start = np.array([forward_dist, 0.0, road_height])
        end = np.array([forward_dist + arrow_length, 0.0, road_height])
        
        pts3d = np.vstack([start, end])
        projected = project_points_safe(pts3d, proj, img.shape)
        
        if len(projected) == 2:
            pt1 = tuple(projected[0].astype(int))
            pt2 = tuple(projected[1].astype(int))
            
            # Draw arrow with thickness proportional to urgency
            thickness = int(3 + (max_length - arrow_length) * 2)  # Thicker when closer to obstacles
            cv2.arrowedLine(img, pt1, pt2, arrow_color, thickness=thickness, tipLength=0.25)
            
            # Also draw a circle at the start for better visibility
            cv2.circle(img, pt1, 5, arrow_color, -1)
            
            # Optional: Draw distance text showing closest obstacle
            if obstacle_clusters:
                closest_dist = get_closest_forward_obstacle_distance(obstacle_clusters)
                if closest_dist < safe_distance:
                    text = f"{closest_dist:.1f}m"
                    cv2.putText(img, text, (pt1[0] + 10, pt1[1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, arrow_color, 2)
            #print("FORWARD DISTANCE:", forward_dist)
            return pt1, pt2, arrow_length
    
    return None, None, arrow_length

def calculate_adaptive_arrow_length(obstacle_clusters, max_length=6.0, min_length=1.0, safe_distance=20.0):
    """
    Calculate arrow length based on proximity of obstacles in the forward direction.
    """
    if not obstacle_clusters:
        return max_length
    
    closest_forward_distance = get_closest_forward_obstacle_distance(obstacle_clusters)
    print(f"Closest forward obstacle distance: {closest_forward_distance:.2f}m")    
    if closest_forward_distance >= safe_distance:
        return max_length
    
    # Linear interpolation between min and max length
    # When distance = 0, length = min_length
    # When distance = safe_distance, length = max_length
    length_ratio = closest_forward_distance / safe_distance
    arrow_length = min_length + (max_length - min_length) * length_ratio
    
    return arrow_length

def get_closest_forward_obstacle_distance(obstacle_clusters, forward_cone_angle=5.0, min_forward_dist=0.5):
    closest_distance = float('inf')

    for cluster in obstacle_clusters:
        if len(cluster) == 0:
            continue

        for pt in cluster:
            if pt[0] < min_forward_dist:
                continue

            angle = np.abs(np.degrees(np.arctan2(pt[1], pt[0])))
            if angle <= forward_cone_angle / 2:
                distance = np.linalg.norm(pt[:2])
                closest_distance = min(closest_distance, distance)

    return closest_distance if closest_distance != float('inf') else float('inf')

def project_points_safe(pts, P, img_shape):
    """Safe projection that handles all edge cases."""
    if len(pts) == 0:
        return np.array([]).reshape(0, 2)
    
    # Convert to homogeneous coordinates
    h = np.hstack([pts, np.ones((len(pts), 1))])
    
    # Project
    uvw = (P @ h.T).T
    z = uvw[:, 2]
    
    # Keep points in front of camera
    valid = z > 0.1
    
    if not np.any(valid):
        return np.array([]).reshape(0, 2)
    
    # Convert to image coordinates
    uv = uvw[valid, :2] / z[valid, np.newaxis]
    
    # Filter points within image bounds (with some margin)
    margin = 10
    img_valid = ((uv[:, 0] >= margin) & (uv[:, 0] < img_shape[1] - margin) & 
                (uv[:, 1] >= margin) & (uv[:, 1] < img_shape[0] - margin))
    
    return uv[img_valid]

##################################################################################################
# —————————————————————————————————————— PIPELINE —————————————————————————————————————— #  

def process_frame_improved(bin_path, args):
    frame = bin_path.stem
    img_path = Path(args.image_dir) / f"{frame}.png"
    calib_path = Path(args.calib_dir) / f"{frame}.txt"
    
    if not (img_path.exists() and calib_path.exists()):
        print(f"[WARN] missing assets for {frame}")
        return
    # Visualization
    proj = parse_calib(calib_path)
    img = cv2.imread(str(img_path))
    # Load and preprocess
    xyz = load_bin(bin_path)
    xyz = xyz[np.abs(xyz[:, 1]) < 10.0]  # Wider lateral crop initially
    xyz = xyz[xyz[:, 0] > 0]  # Only points in front
    
    # Height-based filtering to remove obviously non-ground points
    xyz = xyz[xyz[:, 2] > -3.0]  # Remove points too far below
    xyz = xyz[xyz[:, 2] < 2.0]   # Remove points too far above
    # !!!Enforce deterministic ordering before downsampling
    xyz = xyz[np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))]
    pcd = pc_to_o3d(xyz).voxel_down_sample(0.08)  # Finer voxel grid
    
    # Multi-plane RANSAC
    planes = multi_plane_ransac(pcd, max_planes=3, dist_thresh=0.12)
    
    if not planes:
        print(f"[WARN] No ground planes found for {frame}")
        return
    
    # Combine all plane inliers
    all_ground_indices = []
    for plane, indices in planes:
        all_ground_indices.extend(indices)
    
    candidate_points = np.asarray(pcd.points)[all_ground_indices]
    
    # Adaptive height filtering
    road_candidates = adaptive_height_filtering(candidate_points, grid_size=0.8)
    
    road_points = road_candidates

    # CLUSTER ROAD AND REMOVE SIDEWALK
    
    main_road,left_rough,right_rough =remove_sidewalks(road_points)   

    # debug : to see mask with no curbs removed run the line below                !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #main_road=road_points
    
    # Detect obstacles
    obstacles = detect_obstacles(np.asarray(pcd.points), main_road, 
                               height_threshold=0.4, min_cluster_size=3)
    
    # Cluster obstacles into individual objects
    obstacle_clusters = cluster_obstacles(obstacles, eps=1.2, min_samples=8)
    
    
    
    # Project and draw road points (blue)
    if len(main_road) > 0:
        road_uv = project(main_road, proj)
        for u, v in road_uv:
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (u, v), 2, (255, 100, 0), -1)  # Blue-ish
    
    # Draw obstacle clusters with bounding boxes and distance-based colors
    for i, cluster in enumerate(obstacle_clusters):
        if len(cluster) == 0:
            continue
            
        # Calculate distance from vehicle (assuming vehicle is at origin)
        cluster_center = np.mean(cluster, axis=0)
        distance = np.sqrt(cluster_center[0]**2 + cluster_center[1]**2)
        
        # Get color based on distance
        color = get_distance_color(distance)
        
        # Draw individual obstacle points
        obs_uv = project(cluster, proj)
        for u, v in obs_uv:
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (u, v), 2, color, -1)
        
        # Draw 3D bounding box
        draw_3d_bounding_box(img, cluster, proj, color, thickness=2)
        
        # Draw distance text
        center_2d_points = project_all_points(cluster_center.reshape(1, -1), proj, img.shape)
        if len(center_2d_points) > 0:
            center_2d = center_2d_points[0]
            draw_distance_text(img, center_2d, distance, color)

    # Draw curbs
    if getattr(args, "curbs", False):
        left_uv = project(left_rough, proj)
        right_uv = project(right_rough, proj)

        for u, v in left_uv:
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (u, v), 2, (0, 255, 255), -1)  # Yellow
        
        for u, v in right_uv:
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (u, v), 2, (0, 255, 0), -1)  # Green

    # Add legend
    draw_legend(img)

    print(f"Processed {frame}: {len(main_road)} road points, {len(obstacle_clusters)} obstacle clusters")


    #pt1, pt2 = direction_arrow_adaptive(img, calib_path) # works!!!!
    pt1, pt2,dist = direction_arrow_road_surface(img, calib_path,obstacle_clusters=obstacle_clusters,safe_distance=30) #best
    #print(pt1,pt2)

    #cv2.imshow('Improved Road Detection', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Save image
    script_dir = Path(__file__).parent
    output_dir = script_dir / "B1B2B3"
    output_dir.mkdir(exist_ok=True)
    output_img_path = output_dir / f"{frame}_d_{dist}.png"
    #cv2.imwrite(str(output_img_path), img)
    #print(f"Saved at {output_img_path}")


    return img, dist
    #return main_road, obstacle_clusters

# B1-B2-B3
def run_selected_indexes():
    class Args:
        def __init__(self, velodyne_dir, calib_dir, image_dir, curbs=False):
            self.velodyne_dir = velodyne_dir
            self.calib_dir = calib_dir
            self.image_dir = image_dir
            self.curbs = curbs

    # --- Define your test and train indexes separately
    testing_indexes = ["um_000000", "um_000001","um_000008","um_000009"]
    training_indexes = ["um_000000", "um_000053","um_000023","um_000007","um_000008","um_000043","um_000011","um_000068","um_000025","um_000027","um_000028","um_000030","um_000032","um_000033","um_000037","um_000046","um_000053","um_000003"]
    extra_training= [ "um_000013","um_000001","um_000016","um_000019","um_000021"]

    # --- Define your base paths
    base = "C:/Users/USER/Documents/_CAMERA_LIDAR/data_road"
    
    # --- Testing mode
    testing_args = Args(
        velodyne_dir=f"{base}_velodyne/testing/velodyne",
        calib_dir=f"{base}/testing/calib",
        image_dir=f"{base}/testing/image_2",
        curbs=False
    )
    run_index_list(testing_indexes, testing_args)

    # --- Training mode
    training_args = Args(
        velodyne_dir=f"{base}_velodyne/training/velodyne",
        calib_dir=f"{base}/training/calib",
        image_dir=f"{base}/training/image_2",
        curbs=False
    )
    run_index_list(training_indexes, training_args)

def run_index_list(index_list, args):
    v_dir = Path(args.velodyne_dir)
    for idx in index_list:
        f = v_dir / f"{idx}.bin"
        if f.exists():
            print(f"[INFO] Processing {f.name} from {args.velodyne_dir}")
            img, _ = process_frame_improved(f, args)
            cv2.imshow("Selected Frame Viewer", img)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()  
            if key == 27:  # ESC to exit early
                break
        else:
            print(f"[MISSING] {f}")
    cv2.destroyAllWindows()


# question B4 wall detection mode
def run_wall_test():
    class Args:
        velodyne_dir = "C:\\Users\\USER\\Documents\\GitHub\\Camera_vs_LIDAR\\PART_B\\fakebin"
        calib_dir = "C:\\Users\\USER\\Documents\\GitHub\\Camera_vs_LIDAR\\PART_B\\fakecalib"
        image_dir = "C:\\Users\\USER\\Documents\\GitHub\\Camera_vs_LIDAR\\PART_B\\fakeleft"
        index = "all"  # Or any specific frame ID
        dist = 0.15
        iters = 1000
    
    a = Args()
    v_dir = Path(a.velodyne_dir)
    files = sorted(v_dir.glob('*.bin')) if a.index.lower() == 'all' else [v_dir / f"{a.index}.bin"]
    
    for f in files:
        if f.exists():
            process_frame_improved(f, a)
            # αυτό ξέχασα να το βάλω στο eclass (το show)
            img, _ = process_frame_improved(f, a)
            cv2.imshow("Selected Frame Viewer", img)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()  
            if key == 27:  # ESC to exit early
                break
        else:
            print(f"missing {f}")

def run_interactive(file_list, args):
    for f in file_list:
        if f.exists():
            img, _ = process_frame_improved(f, args)
            cv2.imshow("Improved Road Detection", img)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:  # ESC
                break
        else:
            print(f"[MISSING] {f}")
    cv2.destroyAllWindows()
def run_video_playback(file_list, args, delay=1, save_path="output_video.avi", save_fps=10):
    import os
    import time

    # Ensure output directory exists
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    writer = None
    first_written = False

    for f in file_list:
        if not f.exists():
            print(f"[MISSING] {f}")
            continue

        img, _ = process_frame_improved(f, args)

        # Initialize writer once
        if save_path and not first_written:
            height, width = img.shape[:2]
            writer = cv2.VideoWriter(
                str(save_path),
                cv2.VideoWriter_fourcc(*'XVID'),
                save_fps,
                (width, height)
            )
            first_written = True

        # Save to video file
        if writer:
            writer.write(img)

        # Show live (as fast as possible or fixed delay)
        cv2.imshow("KITTI Live Video", img)
        if cv2.waitKey(delay) == 27:  # ESC
            break

    if writer:
        writer.release()
        print(f"[INFO] Saved video to {save_path}")
    cv2.destroyAllWindows()

# def run_video_playback(file_list, args, delay=100):
#     for f in file_list:
#         if f.exists():
#             img, _ = process_frame_improved(f, args)
#             cv2.imshow("KITTI Video Mode", img)
#             key = cv2.waitKey(delay)
#             if key == 27:  # ESC to break
#                 break
#         else:
#             print(f"[MISSING] {f}")
#     cv2.destroyAllWindows()

def run_video_mode(file_list, args, output_path="output_video.avi"):
    first_img = None
    all_frames = []

    for f in file_list:
        if f.exists():
            print(f"[INFO] Processing {f.name}")
            img, _ = process_frame_improved(f, args)
            if first_img is None:
                first_img = img
            all_frames.append(img)
        else:
            print(f"[WARN] Missing {f}")

    if not all_frames:
        print("[ERROR] No frames to write")
        return

    # Initialize video writer
    height, width = first_img.shape[:2]
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'XVID'),
        10.0,  # FPS
        (width, height)
    )

    for frame in all_frames:
        out.write(frame)

    out.release()
    print(f"[VIDEO SAVED] → {output_path}")

if __name__ == '__main__':
    # erotima B4
    if "--wall" in sys.argv:
        print("Running road detection pipeline for B4: added wall...")
        run_wall_test()
    # debug run for entire kitti dataset
    elif "--kitti" in sys.argv:
        print("Running road detection pipeline for KITTI Dataset...")
        a = get_args()

        # Swap to testing paths if requested
        if a.testing:
            print("Testing mode activated: swapping 'training' → 'testing'")
            a.velodyne_dir = a.velodyne_dir.replace("training", "testing")
            a.calib_dir = a.calib_dir.replace("training", "testing")
            a.image_dir = a.image_dir.replace("training", "testing")

        v_dir = Path(a.velodyne_dir)
        files = sorted(v_dir.glob('*.bin')) if a.index.lower() == 'all' else [v_dir / f"{a.index}.bin"]
        # for f in files:
        #     if f.exists():
        #         process_frame_improved(f, a)
        #     else:
        #         print(f"missing {f}")
        if a.video:
            cur_dir = Path(__file__).parent
            output_dir = cur_dir / "output_video.avi"
            run_video_playback(files, a, delay=1, save_path=output_dir, save_fps=4)
            #run_video_playback(files, a, delay=1)
            
        else:
            run_interactive(files, a)
    # erotima B1B2B3
    else:
        print("Running road detection pipeline: B1-B2-B3...")
        run_selected_indexes()