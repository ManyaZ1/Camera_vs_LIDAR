import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from sidewalk_removed import *

def find_curb_lines(rough_points, road_center_y, side='left', bin_width=0.5, min_points_per_bin=3):
    """
    Find curb line by analyzing height discontinuities in lateral bins
    """
    if len(rough_points) == 0:
        return None
    
    # Filter points to one side
    if side == 'left':
        side_points = rough_points[rough_points[:, 1] < road_center_y]
    else:
        side_points = rough_points[rough_points[:, 1] > road_center_y]
    
    if len(side_points) < min_points_per_bin:
        return None
    
    # Create lateral bins
    y_min, y_max = side_points[:, 1].min(), side_points[:, 1].max()
    n_bins = max(1, int((y_max - y_min) / bin_width))
    
    curb_candidates = []
    
    for i in range(n_bins):
        bin_start = y_min + i * bin_width
        bin_end = bin_start + bin_width
        
        bin_mask = (side_points[:, 1] >= bin_start) & (side_points[:, 1] < bin_end)
        bin_points = side_points[bin_mask]
        
        if len(bin_points) >= min_points_per_bin:
            # Find the point closest to road center in this bin
            if side == 'left':
                closest_idx = np.argmax(bin_points[:, 1])  # rightmost (closest to center)
            else:
                closest_idx = np.argmin(bin_points[:, 1])  # leftmost (closest to center)
            
            curb_candidates.append(bin_points[closest_idx])
    
    if len(curb_candidates) < 2:
        return None
    
    curb_candidates = np.array(curb_candidates)
    
    # Fit line to curb candidates and find the boundary
    curb_y_positions = curb_candidates[:, 1]
    
    # Use median or percentile to find consistent curb position
    if side == 'left':
        curb_boundary = np.percentile(curb_y_positions, 75)  # More conservative
    else:
        curb_boundary = np.percentile(curb_y_positions, 25)
    
    return curb_boundary

def detect_height_transitions(points, road_center_y, side='left', search_radius=1.0, 
                            height_jump_thresh=0.08, lateral_step=0.2):
    """
    Detect height transitions that indicate curb edges
    """
    if len(points) == 0:
        return None
    
    # Filter to one side
    if side == 'left':
        side_mask = points[:, 1] < road_center_y
        start_y = road_center_y
        step_direction = -1
    else:
        side_mask = points[:, 1] > road_center_y
        start_y = road_center_y
        step_direction = 1
    
    side_points = points[side_mask]
    if len(side_points) == 0:
        return None
    
    kdt = cKDTree(side_points[:, :2])
    
    # Step away from road center and look for height jumps
    current_y = start_y
    y_limit = side_points[:, 1].min() if side == 'left' else side_points[:, 1].max()
    
    while abs(current_y - start_y) < abs(y_limit - start_y):
        current_y += step_direction * lateral_step
        
        # Find points near this lateral position
        search_point = np.array([0, current_y])  # Use x=0 as reference
        nearby_indices = kdt.query_ball_point(search_point, search_radius)
        
        if len(nearby_indices) < 3:
            continue
        
        nearby_points = side_points[nearby_indices]
        
        # Check for significant height variation (indicating curb)
        height_range = nearby_points[:, 2].max() - nearby_points[:, 2].min()
        
        if height_range > height_jump_thresh:
            return current_y
    
    return None

def cluster_and_filter_curbs(rough_points, road_center_y, eps=0.8, min_samples=5):
    """
    Cluster rough points and identify which clusters represent actual curbs
    """
    if len(rough_points) == 0:
        return None, None
    
    # Cluster rough points
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(rough_points[:, :2])  # Cluster in 2D
    
    left_curb_y = None
    right_curb_y = None
    
    # Analyze each cluster
    for label in set(labels):
        if label == -1:  # Skip noise
            continue
            
        cluster_points = rough_points[labels == label]
        cluster_center_y = np.median(cluster_points[:, 1])
        
        # Determine if this cluster is on left or right side
        if cluster_center_y < road_center_y:  # Left side
            # Check if this cluster has curb-like properties
            height_std = np.std(cluster_points[:, 2])
            y_spread = cluster_points[:, 1].max() - cluster_points[:, 1].min()
            
            # Good curb clusters have some height variation but are laterally compact
            if height_std > 0.03 and y_spread < 2.0:  # Curb-like cluster
                candidate_y = np.percentile(cluster_points[:, 1], 90)  # Right edge of cluster
                if left_curb_y is None or candidate_y > left_curb_y:  # Closest to road
                    left_curb_y = candidate_y
        
        else:  # Right side
            height_std = np.std(cluster_points[:, 2])
            y_spread = cluster_points[:, 1].max() - cluster_points[:, 1].min()
            
            if height_std > 0.03 and y_spread < 2.0:
                candidate_y = np.percentile(cluster_points[:, 1], 10)  # Left edge of cluster  
                if right_curb_y is None or candidate_y < right_curb_y:  # Closest to road
                    right_curb_y = candidate_y
    
    return left_curb_y, right_curb_y

def adaptive_road_width_estimation(main_road, percentile_range=(25, 75)):
    """
    Estimate typical road width from the main road points
    """
    if len(main_road) == 0:
        return 3.0  # Default road half-width
    
    y_coords = main_road[:, 1]
    road_width = np.percentile(y_coords, percentile_range[1]) - np.percentile(y_coords, percentile_range[0])
    return max(2.0, min(road_width, 8.0))  # Clamp between reasonable values

def robust_sidewalk_removal(main_road, rough_points, min_candidates=30):
    """
    Robustly remove sidewalks using multiple approaches
    """
    if len(main_road) == 0:
        return main_road
    
    road_center_y = np.median(main_road[:, 1])
    road_width = adaptive_road_width_estimation(main_road)
    
    # Method 1: Cluster-based curb detection
    left_curb_cluster, right_curb_cluster = cluster_and_filter_curbs(rough_points, road_center_y)
    
    # Method 2: Height transition detection  
    left_curb_transition = detect_height_transitions(
        np.vstack([main_road, rough_points]), road_center_y, 'left'
    )
    right_curb_transition = detect_height_transitions(
        np.vstack([main_road, rough_points]), road_center_y, 'right'
    )
    
    # Method 3: Line fitting approach
    left_curb_line = find_curb_lines(rough_points, road_center_y, 'left')
    right_curb_line = find_curb_lines(rough_points, road_center_y, 'right')
    
    # Combine methods with fallbacks
    left_candidates = [x for x in [left_curb_cluster, left_curb_transition, left_curb_line] if x is not None]
    right_candidates = [x for x in [right_curb_cluster, right_curb_transition, right_curb_line] if x is not None]
    
    # Choose the most conservative (closest to road center) boundaries
    if left_candidates:
        left_boundary = max(left_candidates)  # Rightmost (most conservative)
    else:
        left_boundary = road_center_y - road_width  # Fallback based on road width
    
    if right_candidates:
        right_boundary = min(right_candidates)  # Leftmost (most conservative)
    else:
        right_boundary = road_center_y + road_width  # Fallback based on road width
    
    # Additional safety checks
    left_rough = rough_points[rough_points[:, 1] < road_center_y] if len(rough_points) > 0 else []
    right_rough = rough_points[rough_points[:, 1] > road_center_y] if len(rough_points) > 0 else []
    
    # Only apply boundary if we have sufficient evidence
    if len(left_rough) < min_candidates:
        left_boundary = -np.inf
    if len(right_rough) < min_candidates:
        right_boundary = np.inf
        
    # Apply boundaries
    road_mask = (main_road[:, 1] > left_boundary) & (main_road[:, 1] < right_boundary)
    filtered_road = main_road[road_mask]
    
    print(f"Road filtering: center_y={road_center_y:.2f}, left_boundary={left_boundary:.2f}, right_boundary={right_boundary:.2f}")
    print(f"Points: {len(main_road)} -> {len(filtered_road)} (removed {len(main_road) - len(filtered_road)})")
    
    return filtered_road

# Updated process_frame_improved function - replace the sidewalk removal section
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
    road_candidates = adaptive_height_filtering(candidate_points, grid_size=0.8)

    road_points=road_candidates

    # Cluster road segments
    road_clusters = cluster_road_segments(road_points, eps=1.5, min_samples=15)
 
   # ... (keep all the existing code until the sidewalk removal section)
    
    # After getting main_road and rough_points:
    if road_clusters:
        main_road = max(road_clusters, key=len)
        main_road = main_road[np.abs(main_road[:, 1]) < 5.0]

        pca_curb_mask = pca_high_variation_mask(main_road, radius=0.3, z_variance_thresh=0.0001)
        rough_points = main_road[pca_curb_mask]
        main_road = main_road[~pca_curb_mask]
        
        # REPLACE the old sidewalk removal with this:
        main_road = robust_sidewalk_removal(main_road, rough_points, min_candidates=50)
        
    else:
        main_road = np.array([]).reshape(0, 3)
        rough_points = np.array([]).reshape(0, 3)
    
    # ... (continue with the rest of the existing code)
    # Detect obstacles
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

    # draw curbs
    rough_uv = project(rough_points, proj)
    for u, v in rough_uv:
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.circle(img, (u, v), 2, (0, 255, 255), -1)  # Yellow

    # # Optional: visualize curb faces (green)
    # if 'curb_points' in locals() and len(curb_points) > 0:
    #     curb_uv = project(curb_points, proj)
    #     for u, v in curb_uv:
    #         if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
    #             cv2.circle(img, (u, v), 2, (0, 255, 0), -1)  # Green

    print(f"Processed {frame}: {len(main_road)} road points, {len(obstacles)} obstacle points")
    cv2.imshow('Improved Road Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #save to directory alt_approach
    #get cuurrent directory
    script_dir= Path(__file__).parent
    #create output directory if not exists
    output_dir = script_dir / "sidewalk"
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
