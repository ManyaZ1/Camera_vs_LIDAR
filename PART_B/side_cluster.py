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
    #p.add_argument("--velodyne_dir", default="C:/Users/USER/Documents/_CAMERA_LIDAR/data_road_velodyne/training/velodyne")
    p.add_argument("--velodyne_dir", default="C:/Users/Mania/Documents/KITTI/data_road_velodyne/training/velodyne")
    p.add_argument("--index", default="all")
    p.add_argument("--dist", type=float, default=0.15)
    p.add_argument("--iters", type=int, default=1000)
    #p.add_argument("--calib_dir", default="C:/Users/USER/Documents/_CAMERA_LIDAR/data_road/training/calib")
    #p.add_argument("--image_dir",  default="C:/Users/USER/Documents/_CAMERA_LIDAR/data_road/training/image_2")
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

#sidewalk
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
def compute_normals(points, radius=0.3):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    return np.asarray(pcd.normals)

def find_curb_by_normals(points, verticality_thresh=0.1):
    '''normal close to 0 means the point is on a curb(vertical surface)'''
    normals = compute_normals(points)
    verticality = np.abs(normals[:, 2])
    curb_mask = verticality < verticality_thresh
    return curb_mask

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

def pca_high_variation_mask(points, radius=0.3, z_variance_thresh=0.002):
    """
    Return a boolean mask of points with high vertical roughness in local patch.
    Useful for excluding curb-like areas from road points.
    """
    kdt = cKDTree(points[:, :2])
    high_variance_mask = np.zeros(len(points), dtype=bool)
    
    for i, p in enumerate(points):
        idx = kdt.query_ball_point(p[:2], radius) 
        #print(f"Point {i}: {len(idx)} neighbors")
        if len(idx) < 5:
            continue
        patch = points[idx]
        pca = PCA(n_components=3).fit(patch)
        #print(f"λ3 = {pca.explained_variance_[2]:.5f}")
        # The 3rd eigenvalue (variance) is largest if there's a jump/step
        if pca.explained_variance_[2] > z_variance_thresh:
            high_variance_mask[i] = True
    return high_variance_mask

# ---------------  parameters you can tune  --------------- #
MIN_CLUSTER_SIZE   = 350            # as before
CURB_BAND_WIDTH_Y  = 6.0            # max lateral distance from road centre [m]
MAX_WALL_HEIGHT_Z  = 1.0            # clusters taller than this are treated as walls
DBSCAN_EPS         = 0.8
DBSCAN_MIN_SAMPLES = 10
# --------------------------------------------------------- #

def curb_side_clusters(side_points, centre_y):
    """Return a list of clusters that are likely to be sidewalk curbs."""
    if len(side_points) < DBSCAN_MIN_SAMPLES:
        return []

    labels = DBSCAN(eps=DBSCAN_EPS,
                    min_samples=DBSCAN_MIN_SAMPLES).fit_predict(side_points[:, :2])

    keep = []
    for lbl in set(labels):
        if lbl == -1:
            continue                          # noise
        mask = labels == lbl
        if mask.sum() < MIN_CLUSTER_SIZE:
            continue                          # too small
        cluster = side_points[mask]

        # 1) close enough to the road centre?
        if np.abs(np.median(cluster[:, 1]) - centre_y) > CURB_BAND_WIDTH_Y:
            continue

        # 2) not a tall wall?
        if cluster[:, 2].ptp() > MAX_WALL_HEIGHT_Z:   # ptp() = max - min
            continue

        keep.append(cluster)

    return keep

MIN_CANDIDATES = 350          # same as you already use
CLUSTER_EPS    = 0.8
MIN_SAMPLES    = 10  



def biggest_cluster(points):
    """Return the largest DBSCAN cluster that is big enough, else None."""
    if len(points) < MIN_SAMPLES:
        return None
    labels = DBSCAN(eps=CLUSTER_EPS, min_samples=MIN_SAMPLES).fit_predict(points[:, :2])
    best, best_len = None, 0
    for lbl in set(labels):
        if lbl == -1:
            continue                      # noise
        mask = labels == lbl
        if mask.sum() >= MIN_CANDIDATES and mask.sum() > best_len:
            best = points[mask]
            best_len = mask.sum()
    return best
from sklearn.cluster import DBSCAN
import numpy as np
import cv2


def curb_side_clusters(
    side_points: np.ndarray,
    centre_y: float,
    *,
    img=None,                 # supply both img & proj if you want on-image debug
    proj=None,
    cluster_color=(0, 255, 255),   # BGR  • left call → yellow, right call → green
    eps: float = 0.8,
    min_samples: int = 10,
    min_cluster_size: int = 350,
    band_width_y: float = 6.0,     # how far from road centre a curb may sit
    max_wall_height_z: float | None = 2.0  # > None ⇒ skip the “tall wall” check
) -> list[np.ndarray]:
    """
    Return **all** clusters that look like sidewalk curbs on ONE side of the road.

    Parameters
    ----------
    side_points : (N,3) ndarray
        Rough curb candidates on either the left or the right of the road.
    centre_y : float
        Median lateral position of the detected road surface.
    img, proj : optional
        If both are given, every accepted cluster is drawn on `img`
        using `cluster_color` (helpful for debugging).
    Other arguments
        Tune only if needed – the defaults match the rest of your pipeline.

    Returns
    -------
    clusters : list[np.ndarray]
        Each entry is an (M,3) array with the XYZ points of one kept cluster.
    """
    # ––– early exit ––––––––––––––––––––––––––––––––––––––––––––––––––– #
    if len(side_points) < min_samples:
        return []

    # ––– DBSCAN clustering in (x,y) –––––––––––––––––––––––––––––––––– #
    labels = DBSCAN(eps=eps,
                    min_samples=min_samples).fit_predict(side_points[:, :2])

    kept: list[np.ndarray] = []

    for lbl in set(labels):
        if lbl == -1:                 # DBSCAN noise
            continue

        mask = labels == lbl
        if mask.sum() < min_cluster_size:
            continue                  # too small → ignore

        cluster = side_points[mask]

        # (1) it must sit not too far from the road centre laterally
        if abs(np.median(cluster[:, 1]) - centre_y) > band_width_y:
            continue

        # (2) it must not be an *entire* tall wall (optional – set None to skip)
        if max_wall_height_z is not None and \
           np.ptp(cluster[:, 2]) > max_wall_height_z:   # np.ptp == max-min
            continue

        kept.append(cluster)

        # ––– OPTIONAL drawing for visual debugging ––––––––––––––– #
        if img is not None and proj is not None:
            uv = project(cluster, proj)                 # your existing helper
            for u, v in uv:
                if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                    cv2.circle(img, (u, v), 2, cluster_color, -1)
        # ––– ----------------------------------------------------- #

    return kept

def curb_side_clusters_old(
    side_points: np.ndarray,
    centre_y: float,
    *,
    img=None,
    proj=None,
    cluster_color=(0, 255, 255),       # default = yellow (BGR)
    eps: float = 0.8,
    min_samples: int = 10,
    min_cluster_size: int = 350,
    band_width_y: float = 6.0,
    max_wall_height_z: float = 1.0,
):
    """
    Identify all sidewalk-curb-like clusters on ONE side of the road.

    Parameters
    ----------
    side_points : (N,3) ndarray
        Rough curb candidates on either the left or the right of the road centre.
    centre_y : float
        Median lateral position of the main road; used to reject far-away walls.
    img, proj : optional
        If both are supplied, each accepted cluster is drawn on `img`
        with `cluster_color` using 2-D projection matrix `proj`.
    cluster_color : tuple(int,int,int)
        BGR colour used when drawing clusters.
    eps, min_samples : DBSCAN parameters
    min_cluster_size : int
        Smallest cluster that is still considered a curb.
    band_width_y : float
        Max. lateral distance (|y_cluster − centre_y|) for a cluster to be counted.
    max_wall_height_z : float
        Clusters with vertical extent greater than this are treated as walls.

    Returns
    -------
    clusters : list[np.ndarray]
        Every element is an (M,3) array representing one accepted curb cluster.
    """
    if len(side_points) < min_samples:
        return []

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(side_points[:, :2])

    kept_clusters = []
    for lbl in set(labels):
        if lbl == -1:                       # DBSCAN noise
            continue
        mask = labels == lbl
        if mask.sum() < min_cluster_size:   # too small → ignore
            continue

        cluster = side_points[mask]

        # (1) close enough laterally to the road centre?
        if abs(np.median(cluster[:, 1]) - centre_y) > band_width_y:
            continue

        # (2) not just a tall wall?
        if np.ptp(cluster[:, 2]) > max_wall_height_z:   # ptp = max - min
            continue

        kept_clusters.append(cluster)

        # ---------- optional visual debugging ----------
        if img is not None and proj is not None:
            uv = project(cluster, proj)                # your existing helper
            for u, v in uv:
                if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                    cv2.circle(img, (u, v), 2, cluster_color, -1)
        # -----------------------------------------------

    return kept_clusters
from sklearn.cluster import DBSCAN

def closest_cluster(points, center_y, eps=0.5, min_samples=10):
    if len(points) == 0:
        return np.empty((0, 3))
    
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, :2]).labels_
    best_cluster = None
    best_dist = float('inf')
    
    for lbl in set(labels):
        if lbl == -1:
            continue
        cluster = points[labels == lbl]
        dist = abs(np.median(cluster[:, 1]) - center_y)
        if dist < best_dist:
            best_dist = dist
            best_cluster = cluster
    
    return best_cluster if best_cluster is not None else np.empty((0, 3))

def process_frame_improved(bin_path, args):
    frame = bin_path.stem
    img_path = Path(args.image_dir) / f"{frame}.png"
    calib_path = Path(args.calib_dir) / f"{frame}.txt"
    
    if not (img_path.exists() and calib_path.exists()):
        print(f"[WARN] missing assets for {frame}")
        return

    # Load and preprocess
    xyz = load_bin(bin_path)
    xyz = xyz[np.abs(xyz[:, 1]) < 10.0]  # Wider lateral crop initially
    xyz = xyz[xyz[:, 0] > 0]  # Only points in front
    
    # Height-based filtering to remove obviously non-ground points
    xyz = xyz[xyz[:, 2] > -3.0]  # Remove points too far below
    xyz = xyz[xyz[:, 2] < 2.0]   # Remove points too far above
    
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
    
    road_points=road_candidates

    # Cluster road segments
    road_clusters = cluster_road_segments(road_points, eps=1.5, min_samples=15)
    
    # Select the largest cluster as the main road
    
    if road_clusters:
        main_road = max(road_clusters, key=len)
        main_road = main_road[np.abs(main_road[:, 1]) < 5.0]

        pca_curb_mask = pca_high_variation_mask(main_road, radius=0.3,  z_variance_thresh=0.0001) #radius=.5
        rough_points  = main_road[pca_curb_mask]      # yellow
        main_road     = main_road[~pca_curb_mask]     # blue
    else:
        main_road = np.array([]).reshape(0, 3)
        rough_points = np.array([]).reshape(0, 3)
    proj = parse_calib(calib_path)
    img = cv2.imread(str(img_path))

    # -------------  left / right rough split  ------------- #
    road_center_y = np.median(main_road[:, 1])

    left_rough   = rough_points[ rough_points[:, 1] < road_center_y ]
    right_rough  = rough_points[ rough_points[:, 1] > road_center_y ]

    left_cluster  = closest_cluster(left_rough,  road_center_y)
    right_cluster = closest_cluster(right_rough, road_center_y)

    if len(left_cluster) > MIN_CANDIDATES:
        left_curb_y = np.percentile(left_cluster[:, 1], 95)
    else:
        left_curb_y = -np.inf

    if len(right_cluster) > MIN_CANDIDATES:
        right_curb_y = np.percentile(right_cluster[:, 1], 5)
    else:
        right_curb_y = np.inf

    main_road = main_road[(main_road[:, 1] > left_curb_y) & (main_road[:, 1] < right_curb_y)]





    obstacles = detect_obstacles(np.asarray(pcd.points), main_road, 
                               height_threshold=0.4, min_cluster_size=3)
    
    # Visualization
    proj = parse_calib(calib_path)
    img = cv2.imread(str(img_path))
    
    # Project and draw road points (blue)
    if len(main_road) > 0:
        road_uv = project(main_road, proj)
        for u, v in road_uv:
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (u, v), 2, (255, 100, 0), -1)  # Blue-ish
    
    # Project and draw obstacles (red)
    if len(obstacles) > 0:
        obs_uv = project(obstacles, proj)
        for u, v in obs_uv:
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (u, v), 3, (0, 0, 255), -1)  # Red

   

    print(f"Processed {frame}: {len(main_road)} road points, {len(obstacles)} obstacle points")
    cv2.imshow('Improved Road Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #save to directory alt_approach
    #get cuurrent directory
    script_dir= Path(__file__).parent
    #create output directory if not exists
    output_dir = script_dir / "sidewalk_removal"
    output_dir.mkdir(exist_ok=True)
    #save image
    output_img_path = output_dir / f"{frame}_road_detection.png"
    #cv2.imwrite(str(output_img_path), img)
    print(f"saved at {output_img_path}")
    
    return main_road, obstacles

# Helper functions (keep your existing ones)
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
