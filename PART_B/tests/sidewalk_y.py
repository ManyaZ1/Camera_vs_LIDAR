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
from sklearn.cluster import DBSCAN

MIN_CANDIDATES = 350   # όριο για «έγκυρο» cluster

def curb_y_from_clusters(points, side, eps=0.8, min_samples=10):
    """
    side: 'left'  ή  'right'
    Επιστρέφει ένα single y-όριο (curb_y) προερχόμενο από ΟΛΑ τα
    DBSCAN-clusters που έχουν ≥ MIN_CANDIDATES σημεία.
    - left  →   παίρνουμε το max 95-percentile (πιο κοντά στο κέντρο)
    - right →   παίρνουμε το min  5-percentile (πιο κοντά στο κέντρο)
    """
    if len(points) < min_samples:
        return -np.inf if side == 'left' else np.inf

    lbl = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points[:, :2])
    percs = []
    for l in set(lbl):
        if l == -1:               # θόρυβος
            continue
        cl = points[lbl == l]
        if len(cl) < MIN_CANDIDATES:
            continue
        if side == 'left':
            percs.append(np.percentile(cl[:, 1], 95))  # πιο «δεξιά» τιμή
        else:
            percs.append(np.percentile(cl[:, 1], 5))   # πιο «αριστερά» τιμή

    if not percs:                      # κανένα cluster δεν πληροί το size
        return -np.inf if side == 'left' else np.inf
    return max(percs) if side == 'left' else min(percs)

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
def get_y_ranges_of_clusters(points, eps=0.8, min_samples=10, min_cluster_size=350):
    if len(points) < min_samples:
        return []

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, :2])
    labels = db.labels_
    y_ranges = []

    for lbl in set(labels):
        if lbl == -1:
            continue
        cluster = points[labels == lbl]
        if len(cluster) >= min_cluster_size:
            y_min, y_max = cluster[:, 1].min(), cluster[:, 1].max()
            y_ranges.append((y_min, y_max))

    return y_ranges
def exclude_by_y_ranges(points, y_ranges):
    mask = np.ones(len(points), dtype=bool)
    for y_min, y_max in y_ranges:
        in_range = (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        mask &= ~in_range  # remove these
    return points[mask]
def process_frame_improved(bin_path, args):
    # ---------- paths ----------
    frame      = bin_path.stem
    img_path   = Path(args.image_dir) / f"{frame}.png"
    calib_path = Path(args.calib_dir) / f"{frame}.txt"
    if not (img_path.exists() and calib_path.exists()):
        print(f"[WARN] assets missing for {frame}"); return

    # ---------- load / pre-crop ----------
    xyz = load_bin(bin_path)
    xyz = xyz[(xyz[:, 0] > 0) & (xyz[:, 2] > -3.0) & (xyz[:, 2] < 2.0)
              & (np.abs(xyz[:, 1]) < 10.0)]

    pcd    = pc_to_o3d(xyz).voxel_down_sample(0.08)
    planes = multi_plane_ransac(pcd, max_planes=3, dist_thresh=0.12)
    if not planes:
        print(f"[WARN] no planes for {frame}"); return
    ground_idx = [i for _, idx in planes for i in idx]
    candidates = np.asarray(pcd.points)[ground_idx]

    # ---------- height filter & initial clusters ----------
    road_candidates = adaptive_height_filtering(candidates, grid_size=0.8)
    clusters        = cluster_road_segments(road_candidates, eps=1.5, min_samples=15)
    if not clusters:
        print(f"[WARN] no road cluster for {frame}"); return
    main_road = max(clusters, key=len)
    main_road = main_road[np.abs(main_road[:, 1]) < 5.0]

    # ---------- curb / rough split ----------
    rough_mask  = pca_high_variation_mask(main_road, radius=0.3, z_variance_thresh=1e-4)
    rough_pts   = main_road[rough_mask]          # potential curb (yellow/green)
    # split rough σε left / right
    med_y       = np.median(main_road[:, 1])
    left_rough  = rough_pts[ rough_pts[:, 1] < med_y]
    right_rough = rough_pts[ rough_pts[:, 1] > med_y]

    # ----- ΝΕΟ: curb-y με πολλά clusters
    left_curb_y  = curb_y_from_clusters(left_rough,  side='left')
    right_curb_y = curb_y_from_clusters(right_rough, side='right')

    # τελικό φιλτράρισμα road – ίδιο λογικό masking όπως παλιά
    main_road = main_road[(main_road[:, 1] > left_curb_y) &
                        (main_road[:, 1] < right_curb_y)]

    # ---------- helper: valid DBSCAN clusters ----------
    from sklearn.cluster import DBSCAN
    MIN_CANDIDATES = 350
    def db_clusters(pts, eps=0.8, min_samples=10, min_size=MIN_CANDIDATES):
        if len(pts) < min_samples: return []
        lbl = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts[:, :2])
        keep = []
        for l in set(lbl):
            if l == -1: continue
            cl = pts[lbl == l]
            if len(cl) >= min_size:
                keep.append(cl)
        return keep

    left_cls   = db_clusters(left_rough)
    right_cls  = db_clusters(right_rough)

    # ---------- depth-based (x) exclusion ----------
    DEPTH_TOL = 0.40         # metres around each curb depth
    def excl_depth(road, cls_list, tol=DEPTH_TOL):
        if not cls_list: return road
        centers = np.array([np.mean(cl[:, 0]) for cl in cls_list])  # x-centres
        mask = np.ones(len(road), bool)
        for cx in centers:
            mask &= np.abs(road[:, 0] - cx) > tol
        return road[mask]

    main_road = excl_depth(main_road, left_cls + right_cls)

    # ---------- obstacle detection ----------
    obstacles = detect_obstacles(np.asarray(pcd.points),
                                 main_road, height_threshold=0.4)

    # ---------- visualisation ----------
    proj = parse_calib(calib_path)
    img  = cv2.imread(str(img_path))
    for u, v in project(main_road,  proj): cv2.circle(img,(u,v),2,(255,100,0),-1)
    for u, v in project(obstacles,  proj): cv2.circle(img,(u,v),3,(0,0,255),-1)
    for u, v in project(left_rough, proj): cv2.circle(img,(u,v),2,(0,255,255),-1)
    for u, v in project(right_rough,proj): cv2.circle(img,(u,v),2,(0,255,0),-1)

    print(f"{frame}: road={len(main_road)}, obst={len(obstacles)}")
    cv2.imshow("Improved Road Detection", img)
    cv2.waitKey(0); cv2.destroyAllWindows()

    outdir = Path(__file__).parent / "sidewalk"
    outdir.mkdir(exist_ok=True)
    cv2.imwrite(str(outdir / f"{frame}_road_detection.png"), img)


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
