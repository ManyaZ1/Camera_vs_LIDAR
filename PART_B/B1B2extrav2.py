import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import cv2
from pathlib import Path
import argparse
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree

LOCAL_HEIGHT_TRESHOLD = 0.1 # adaptive_height_filtering
HEIGHT_VARIATION_THRESHOLD = 0.1  # road_continuity_filter

def get_args():
    p = argparse.ArgumentParser("KITTI Velodyne viewer + road RANSAC")
    p.add_argument("--velodyne_dir", default="C:/Users/Mania/Documents/KITTI/data_road_velodyne/training/velodyne")
    p.add_argument("--index", default="all")
    p.add_argument("--dist", type=float, default=0.15)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--calib_dir", default="C:/Users/Mania/Documents/KITTI/data_road/training/calib")
    p.add_argument("--image_dir",  default="C:/Users/Mania/Documents/KITTI/data_road/training/image_2")
    return p.parse_args()

def pc_to_o3d(xyz):
    '''Convert numpy array to Open3D PointCloud'''
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    return pc

def multi_plane_ransac(pcd, max_planes=3, dist_thresh=0.1, min_points=100):
    '''Find multiple ground planes using iterative RANSAC'''
    points = np.asarray(pcd.points)
    remaining_indices = set(range(len(points)))
    planes = []
    
    for _ in range(max_planes):
        if len(remaining_indices) < min_points:
            break
            
        sub_indices = list(remaining_indices)
        sub_pcd = o3d.geometry.PointCloud()
        sub_pcd.points = o3d.utility.Vector3dVector(points[sub_indices])
        
        plane, inliers = sub_pcd.segment_plane( 
            distance_threshold=dist_thresh, 
            ransac_n=3, 
            num_iterations=1000
        )
        
        if len(inliers) < min_points:
            break
        
        original_inliers = [sub_indices[i] for i in inliers]
        planes.append((plane, original_inliers))
        remaining_indices -= set(original_inliers)
    
    return planes

def adaptive_height_filtering(points, grid_size=1.0, percentile=10):
    '''Adaptive height filtering based on local ground level'''
    if len(points) == 0:
        return np.array([]).reshape(0, 3)
    
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    x_bins = int((x_max - x_min) / grid_size) + 1
    y_bins = int((y_max - y_min) / grid_size) + 1
    
    x_idx = ((points[:, 0] - x_min) / grid_size).astype(int)
    y_idx = ((points[:, 1] - y_min) / grid_size).astype(int)
    
    x_idx = np.clip(x_idx, 0, x_bins - 1)
    y_idx = np.clip(y_idx, 0, y_bins - 1)
    
    ground_levels = np.full((x_bins, y_bins), np.nan)
    
    for i in range(x_bins):
        for j in range(y_bins):
            mask = (x_idx == i) & (y_idx == j)
            if np.sum(mask) > 0:
                ground_levels[i, j] = np.percentile(points[mask, 2], percentile)
    
    from scipy.interpolate import griddata
    valid_mask = ~np.isnan(ground_levels)
    if np.sum(valid_mask) > 3:
        xi, yi = np.meshgrid(range(x_bins), range(y_bins), indexing='ij')
        valid_points = np.column_stack([xi[valid_mask], yi[valid_mask]])
        valid_values = ground_levels[valid_mask]
        
        all_points = np.column_stack([xi.ravel(), yi.ravel()])
        interpolated = griddata(valid_points, valid_values, all_points, method='linear')
        ground_levels = interpolated.reshape(x_bins, y_bins)
    
    filtered_indices = []
    for idx, (x, y, z) in enumerate(points):
        local_ground = ground_levels[x_idx[idx], y_idx[idx]]
        if not np.isnan(local_ground) and (z - local_ground) < LOCAL_HEIGHT_TRESHOLD:
            filtered_indices.append(idx)
    
    return points[filtered_indices]

def road_continuity_filter(points, search_radius=2.0, min_neighbors=5):
    '''Filter points based on road continuity'''
    if len(points) < min_neighbors:
        return points
    
    tree = cKDTree(points[:, :2])
    valid_indices = []
    
    for i, point in enumerate(points):
        neighbors = tree.query_ball_point(point[:2], search_radius)
        
        if len(neighbors) >= min_neighbors:
            neighbor_points = points[neighbors]
            height_std = np.std(neighbor_points[:, 2])
            if height_std < HEIGHT_VARIATION_THRESHOLD:
                valid_indices.append(i)
    
    return points[valid_indices]

def cluster_road_segments(points, eps=1.0, min_samples=10):
    '''Cluster road points into coherent segments'''
    if len(points) < min_samples:
        return [points]
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(points[:, :2])
    
    clusters = []
    for label in set(labels):
        if label != -1:
            cluster_points = points[labels == label]
            if len(cluster_points) >= min_samples:
                clusters.append(cluster_points)
    
    return clusters

def improved_obstacle_detection(all_points, road_points, height_threshold=0.2, 
                              min_cluster_size=8, max_cluster_size=3000, max_distance=40.0):
    '''Improved obstacle detection with better filtering and clustering'''
    if len(road_points) == 0:
        return []
    
    # Filter points by distance to focus on relevant objects
    distances = np.linalg.norm(all_points[:, :2], axis=1)
    nearby_points = all_points[distances < max_distance]
    
    # Create spatial index for road surface
    road_tree = cKDTree(road_points[:, :2])
    
    # Find potential obstacle points with lower threshold
    obstacle_candidates = []
    for point in nearby_points:
        # Find nearest road points for height estimation
        distances, indices = road_tree.query(point[:2], k=min(5, len(road_points)))
        
        if len(indices) > 0 and distances[0] < 5.0:  # Only if close to road
            # Use simple average for height estimation
            local_road_height = np.mean(road_points[indices, 2])
            
            # Lower threshold for obstacle detection
            if point[2] - local_road_height > height_threshold:
                obstacle_candidates.append(point)
    
    if len(obstacle_candidates) == 0:
        return []
    
    obstacle_candidates = np.array(obstacle_candidates)
    
    # More aggressive clustering parameters for sparse LIDAR data
    clustering = DBSCAN(eps=0.8, min_samples=5)  # Increased eps, reduced min_samples
    labels = clustering.fit_predict(obstacle_candidates)
    
    obstacle_clusters = []
    for label in set(labels):
        if label != -1:  # Ignore noise
            cluster_points = obstacle_candidates[labels == label]
            
            # More permissive cluster size filtering
            if min_cluster_size <= len(cluster_points) <= max_cluster_size:
                obstacle_clusters.append(cluster_points)
    
    return obstacle_clusters

def classify_obstacle_clusters(clusters, road_points):
    '''Classify obstacle clusters based on geometric features'''
    classified_objects = []
    
    for cluster in clusters:
        if len(cluster) < 5:
            continue
            
        # Extract geometric features
        features = extract_geometric_features(cluster, road_points)
        
        # Simple rule-based classification
        obj_type = classify_by_features(features, cluster)
        
        classified_objects.append({
            'points': cluster,
            'type': obj_type,
            'features': features,
            'bbox': compute_3d_bbox(cluster)
        })
    
    return classified_objects

def extract_geometric_features(cluster, road_points):
    '''Extract geometric features for classification'''
    # Basic dimensions
    min_coords = np.min(cluster, axis=0)
    max_coords = np.max(cluster, axis=0)
    dimensions = max_coords - min_coords
    
    # Height above road
    road_tree = cKDTree(road_points[:, :2])
    center_2d = np.mean(cluster[:, :2], axis=0)
    _, indices = road_tree.query(center_2d, k=min(10, len(road_points)))
    local_road_height = np.mean(road_points[indices, 2])
    height_above_road = np.mean(cluster[:, 2]) - local_road_height
    
    # Shape analysis using PCA
    pca = PCA(n_components=3)
    pca.fit(cluster)
    eigenvalues = pca.explained_variance_
    
    # Ratios for shape analysis
    ratio_1_2 = eigenvalues[0] / (eigenvalues[1] + 1e-6)
    ratio_2_3 = eigenvalues[1] / (eigenvalues[2] + 1e-6)
    
    # Density
    volume = np.prod(dimensions)
    density = len(cluster) / (volume + 1e-6)
    
    return {
        'dimensions': dimensions,
        'height_above_road': height_above_road,
        'eigenvalue_ratios': [ratio_1_2, ratio_2_3],
        'density': density,
        'point_count': len(cluster),
        'volume': volume
    }

def classify_by_features(features, cluster):
    '''Rule-based classification using geometric features with more relaxed thresholds'''
    dims = features['dimensions']
    height = features['height_above_road']
    point_count = features['point_count']
    ratios = features['eigenvalue_ratios']
    volume = features['volume']
    
    # Debug output
    print(f"Classifying object: dims=({dims[0]:.2f}, {dims[1]:.2f}, {dims[2]:.2f}), height={height:.2f}, points={point_count}, ratios={ratios}")
    
    # Large vehicles first (trucks, buses) - most restrictive
    if (dims[0] > 6.0 or dims[1] > 3.0 or 
        (height > 3.0 and point_count > 200) or
        volume > 30.0):
        return 'large_vehicle'
    
    # Car-like objects - much more relaxed thresholds
    elif (((dims[0] > 2.0 and dims[1] > 1.0) or  # Minimum car-like dimensions
           (dims[0] > 1.5 and dims[1] > 1.2)) and  # Alternative threshold
          height > 0.8 and height < 3.0 and        # Reasonable height range
          point_count >= 15 and                     # Minimum points for a car
          volume > 2.0 and volume < 25.0):          # Reasonable volume
        return 'car'
    
    # Pole-like objects (traffic signs, poles) - check before pedestrian
    elif (max(dims[0], dims[1]) < 0.8 and         # Thin in horizontal plane
          height > 1.5 and                         # Tall
          ratios[0] > 5.0 and                      # Elongated shape
          point_count > 10):
        return 'pole'
    
    # Pedestrian-like objects
    elif (max(dims[0], dims[1]) < 1.5 and         # Small horizontal footprint
          min(dims[0], dims[1]) > 0.2 and         # Not too thin
          height > 1.2 and height < 2.5 and       # Human height range
          point_count >= 8 and point_count < 150 and  # Reasonable point count
          volume < 3.0):                          # Small volume
        return 'pedestrian'
    
    # Cyclist-like objects
    elif (((dims[0] > 1.2 and dims[1] < 1.5) or   # Bike-like proportions
           (dims[1] > 1.2 and dims[0] < 1.5)) and
          height > 0.8 and height < 2.5 and
          point_count >= 10 and point_count < 200 and
          volume > 0.5 and volume < 8.0):
        return 'cyclist'
    
    # Small objects (might be debris, small obstacles)
    elif (max(dims) < 1.0 and height < 1.0 and point_count < 50):
        return 'small_object'
    
    else:
        return 'unknown'

def compute_3d_bbox(points):
    '''Compute 3D bounding box for visualization'''
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    center = (min_coords + max_coords) / 2
    size = max_coords - min_coords
    
    return {
        'center': center,
        'size': size,
        'min': min_coords,
        'max': max_coords
    }

def draw_3d_bbox_on_image(img, bbox, projection_matrix, color=(0, 255, 0)):
    '''Draw 3D bounding box on image'''
    center = bbox['center']
    size = bbox['size']
    
    # Define 8 corners of the 3D bounding box
    corners_3d = np.array([
        [center[0] - size[0]/2, center[1] - size[1]/2, center[2] - size[2]/2],
        [center[0] + size[0]/2, center[1] - size[1]/2, center[2] - size[2]/2],
        [center[0] + size[0]/2, center[1] + size[1]/2, center[2] - size[2]/2],
        [center[0] - size[0]/2, center[1] + size[1]/2, center[2] - size[2]/2],
        [center[0] - size[0]/2, center[1] - size[1]/2, center[2] + size[2]/2],
        [center[0] + size[0]/2, center[1] - size[1]/2, center[2] + size[2]/2],
        [center[0] + size[0]/2, center[1] + size[1]/2, center[2] + size[2]/2],
        [center[0] - size[0]/2, center[1] + size[1]/2, center[2] + size[2]/2],
    ])
    
    # Project to image coordinates
    corners_2d = project(corners_3d, projection_matrix)
    
    if len(corners_2d) == 8:
        # Draw the 12 edges of the bounding box
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]
        
        for edge in edges:
            pt1 = tuple(corners_2d[edge[0]])
            pt2 = tuple(corners_2d[edge[1]])
            cv2.line(img, pt1, pt2, color, 2)

def filter_clusters_by_size(points, eps=0.8, min_samples=10, min_cluster_size=350):
    if len(points) < min_samples:
        return []
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, :2])
    labels = db.labels_
    clusters = []
    for lbl in set(labels):
        if lbl == -1:
            continue
        cluster = points[labels == lbl]
        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)
    return clusters

def pca_high_variation_mask(points, radius=0.3, z_variance_thresh=0.002):
    '''Return a boolean mask of points with high vertical roughness in local patch'''
    kdt = cKDTree(points[:, :2])
    high_variance_mask = np.zeros(len(points), dtype=bool)
    
    for i, p in enumerate(points):
        idx = kdt.query_ball_point(p[:2], radius) 
        if len(idx) < 5:
            continue
        patch = points[idx]
        pca = PCA(n_components=3).fit(patch)
        if pca.explained_variance_[2] > z_variance_thresh:
            high_variance_mask[i] = True
    return high_variance_mask

def process_frame_improved(bin_path, args):
    frame = bin_path.stem
    img_path = Path(args.image_dir) / f"{frame}.png"
    calib_path = Path(args.calib_dir) / f"{frame}.txt"
    
    if not (img_path.exists() and calib_path.exists()):
        print(f"[WARN] missing assets for {frame}")
        return

    # Load and preprocess
    xyz = load_bin(bin_path)
    xyz = xyz[np.abs(xyz[:, 1]) < 15.0]  # Wider lateral range for better detection
    xyz = xyz[xyz[:, 0] > 0]  # Only points in front
    xyz = xyz[xyz[:, 0] < 50.0]  # Limit to reasonable distance
    
    # Height-based filtering
    xyz = xyz[xyz[:, 2] > -3.0]
    xyz = xyz[xyz[:, 2] < 5.0]   # Allow higher objects
    
    pcd = pc_to_o3d(xyz).voxel_down_sample(0.08)
    
    # Multi-plane RANSAC for ground detection
    planes = multi_plane_ransac(pcd, max_planes=3, dist_thresh=0.12)
    
    if not planes:
        print(f"[WARN] No ground planes found for {frame}")
        return
    
    # Combine all plane inliers
    all_ground_indices = []
    for plane, indices in planes:
        all_ground_indices.extend(indices)
    
    candidate_points = np.asarray(pcd.points)[all_ground_indices]
    road_candidates = adaptive_height_filtering(candidate_points, grid_size=0.8)
    road_points = road_candidates
    
    # Cluster road segments and get main road
    road_clusters = cluster_road_segments(road_points, eps=1.5, min_samples=15)
    
    if road_clusters:
        main_road = max(road_clusters, key=len)
        main_road = main_road[np.abs(main_road[:, 1]) < 8.0]
        
        # Curb detection for road boundary refinement
        pca_curb_mask = pca_high_variation_mask(main_road, radius=0.3, z_variance_thresh=0.0001)
        rough_points = main_road[pca_curb_mask]
        main_road = main_road[~pca_curb_mask]
        
        # Refine road boundaries using curb information
        road_center_y = np.median(main_road[:, 1])
        left_rough = rough_points[rough_points[:, 1] < road_center_y]
        right_rough = rough_points[rough_points[:, 1] > road_center_y]
        
        MIN_CANDIDATES = 300
        
        if len(left_rough) > MIN_CANDIDATES:
            left_curb_y = np.percentile(left_rough[:, 1], 95)
        else:
            left_curb_y = -np.inf
            
        if len(right_rough) > MIN_CANDIDATES:
            right_curb_y = np.percentile(right_rough[:, 1], 5)
        else:
            right_curb_y = np.inf
        
        main_road = main_road[(main_road[:, 1] > left_curb_y) & (main_road[:, 1] < right_curb_y)]
    else:
        main_road = np.array([]).reshape(0, 3)
        left_rough = np.array([]).reshape(0, 3)
        right_rough = np.array([]).reshape(0, 3)
    
    # Improved obstacle detection and classification
    obstacle_clusters = improved_obstacle_detection(
        np.asarray(pcd.points), main_road, 
        height_threshold=0.3, min_cluster_size=10, max_cluster_size=2000
    )
    
    classified_objects = classify_obstacle_clusters(obstacle_clusters, main_road)
    
    # Visualization
    proj = parse_calib(calib_path)
    img = cv2.imread(str(img_path))
    
    # Project and draw road points (blue)
    if len(main_road) > 0:
        road_uv = project(main_road, proj)
        for u, v in road_uv:
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (u, v), 2, (255, 100, 0), -1)
    
    # Draw classified objects with different colors and bounding boxes
    colors = {
        'car': (0, 0, 255),           # Red
        'pedestrian': (255, 0, 0),    # Blue
        'cyclist': (0, 255, 255),     # Yellow
        'pole': (255, 0, 255),        # Magenta
        'large_vehicle': (0, 128, 255), # Orange
        'unknown': (128, 128, 128)    # Gray
    }
    
    for obj in classified_objects:
        obj_type = obj['type']
        points = obj['points']
        bbox = obj['bbox']
        color = colors.get(obj_type, (128, 128, 128))
        
        # Draw points
        obj_uv = project(points, proj)
        for u, v in obj_uv:
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (u, v), 3, color, -1)
        
        # Draw 3D bounding box
        draw_3d_bbox_on_image(img, bbox, proj, color)
        
        # Add label
        center_2d = project(bbox['center'].reshape(1, -1), proj)
        if len(center_2d) > 0:
            cv2.putText(img, obj_type, tuple(center_2d[0]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw curbs
    # if len(left_rough) > 0:
    #     left_uv = project(left_rough, proj)
    #     for u, v in left_uv:
    #         if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
    #             cv2.circle(img, (u, v), 2, (0, 255, 255), -1)  # Yellow
    
    # if len(right_rough) > 0:
    #     right_uv = project(right_rough, proj)
    #     for u, v in right_uv:
    #         if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
    #             cv2.circle(img, (u, v), 2, (0, 255, 0), -1)  # Green
    
    print(f"Processed {frame}: {len(main_road)} road points, {len(classified_objects)} objects")
    for obj in classified_objects:
        print(f"  - {obj['type']}: {len(obj['points'])} points")
    
    cv2.imshow('Improved Object Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save results
    script_dir = Path(__file__).parent
    output_dir = script_dir / "B1B2_improved"
    output_dir.mkdir(exist_ok=True)
    output_img_path = output_dir / f"{frame}_classified.png"
    cv2.imwrite(str(output_img_path), img)
    print(f"Saved at {output_img_path}")
    
    return main_road, classified_objects

# Helper functions
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

if __name__ == '__main__':
    a = get_args()
    v_dir = Path(a.velodyne_dir)
    files = sorted(v_dir.glob('*.bin')) if a.index.lower() == 'all' else [v_dir / f"{a.index}.bin"]
    for f in files:
        if f.exists():
            process_frame_improved(f, a)
        else:
            print(f"missing {f}")