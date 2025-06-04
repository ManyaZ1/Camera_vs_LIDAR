import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import cv2
from pathlib import Path
import argparse
LOCAL_HEIGHT_TRESHOLD = 0.1 # adaptive_height_filtering
HEIGHT_VARIATION_THRESHOLD = 0.1  # road_continuity_filter

def get_args():
    p = argparse.ArgumentParser("KITTI Velodyne viewer + road RANSAC")
    p.add_argument("--velodyne_dir", default="C:/Users/USER/Documents/_CAMERA_LIDAR/data_road_velodyne/training/velodyne")
    p.add_argument("--index", default="all")
    p.add_argument("--dist", type=float, default=0.15)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--calib_dir", default="C:/Users/USER/Documents/_CAMERA_LIDAR/data_road/training/calib")
    p.add_argument("--image_dir",  default="C:/Users/USER/Documents/_CAMERA_LIDAR/data_road/training/image_2")
    return p.parse_args()


def pc_to_o3d(xyz):
    '''Convert numpy array to Open3D PointCloud'''
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    return pc

def ransac_ground(pcd, dist, iters):
    '''Run RANSAC to find ground plane in point cloud'''
    plane, inl = pcd.segment_plane(distance_threshold=dist, ransac_n=3, num_iterations=iters)
    refined = refine_inliers_by_distance(np.asarray(pcd.points), plane, inl)
    return plane, list(refined), list(inl)

def refine_inliers_by_distance(points, plane, idx):
    '''Refine inliers by signed distance to the plane'''
    a, b, c, d = plane
    normal = np.array([a, b, c])
    denom = np.linalg.norm(normal)
    signed = (points @ normal + d) / denom
    keep = (signed >= -0.15) & (signed <= 0)
    return [i for i in idx if keep[i]]

def filter_points_by_local_smoothness(pts, radius=0.25, height_thresh=0.08):
    if len(pts) == 0:
        return pts
    tree = cKDTree(pts[:, :2])
    good = []
    for i, p in enumerate(pts):
        nbr = tree.query_ball_point(p[:2], radius)
        if len(nbr) < 3:
            continue
        if np.std(pts[nbr][:, 2]) < height_thresh:
            good.append(i)
    return pts[good]

def remove_sidewalks_balanced(points, road_width=6.0, sidewalk_gap=1.0):
    '''Remove sidewalks while preserving both lanes symmetrically'''
    if len(points) == 0:
        return points
    
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    section_length = 2.0  # 2 meter sections
    
    filtered_points = []
    
    for x_start in np.arange(x_min, x_max, section_length):
        x_end = x_start + section_length
        section_mask = (points[:, 0] >= x_start) & (points[:, 0] < x_end)
        section_points = points[section_mask]
        
        if len(section_points) < 5:
            continue
        
        # Analyze lateral distribution
        y_coords = section_points[:, 1]
        y_min, y_max = y_coords.min(), y_coords.max()
        total_width = y_max - y_min
        
        # If section is very wide, likely includes sidewalks
        if total_width > road_width + 2 * sidewalk_gap:
            # Find the densest central region
            y_center = np.median(y_coords)  # Use median as robust center
            
            # Create histogram to find density peaks
            hist, bins = np.histogram(y_coords, bins=min(20, len(section_points)//3))
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # Find the peak closest to the median center
            center_distances = np.abs(bin_centers - y_center)
            peak_idx = np.argmax(hist)  # Highest density
            
            # Alternative: find peak closest to center if main peak is off-center
            if center_distances[peak_idx] > road_width / 2:
                center_peak_candidates = np.where(hist > hist.max() * 0.5)[0]  # Peaks with >50% of max density
                if len(center_peak_candidates) > 0:
                    center_distances_candidates = center_distances[center_peak_candidates]
                    peak_idx = center_peak_candidates[np.argmin(center_distances_candidates)]
            
            # Keep points around the identified peak
            peak_center = bin_centers[peak_idx]
            road_mask = np.abs(y_coords - peak_center) <= road_width / 2
            
            filtered_points.append(section_points[road_mask])
        else:
            # Section is reasonable width, keep all points
            filtered_points.append(section_points)
    
    if filtered_points:
        return np.vstack(filtered_points)
    else:
        return points

def remove_isolated_regions(points, min_cluster_size=20, max_clusters=3):
    '''Remove small isolated regions, but preserve main road clusters'''
    if len(points) < min_cluster_size:
        return points
    
    # Simple clustering based on 2D distance
    clustering = DBSCAN(eps=2.0, min_samples=8)  # More lenient clustering
    labels = clustering.fit_predict(points[:, :2])
    
    # Keep clusters with sufficient points
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    
    if len(unique_labels) == 0:
        return points
    
    # Keep larger clusters
    valid_labels = unique_labels[counts >= min_cluster_size]
    
    if len(valid_labels) == 0:
        return points
    
    # If too many clusters, keep the largest ones
    if len(valid_labels) > max_clusters:
        # Sort by size and keep largest
        sorted_indices = np.argsort(counts[np.isin(unique_labels, valid_labels)])[-max_clusters:]
        valid_labels = valid_labels[sorted_indices]
    
    # Return points from valid clusters
    valid_mask = np.isin(labels, valid_labels)
    return points[valid_mask]

def filter_largest_connected_component(uv: np.ndarray, img_shape: tuple) -> list:
    """
    Keeps only the (u, v) points that belong to the largest connected component.
    """
    H, W = img_shape[:2]
    mask_img = np.zeros((H, W), dtype=np.uint8)

    for u, v in uv:
        if 0 <= u < W and 0 <= v < H:
            mask_img[v, u] = 255

    # Morphological smoothing to connect nearby points
    mask_img = cv2.dilate(mask_img, np.ones((5, 5), np.uint8), iterations=1)
    mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    # Connected components
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_img)

    if nlabels <= 1:
        return []

    # Find the largest component (excluding background label 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # Keep only UV points in largest component
    final_uv = [(u, v) for (u, v) in uv if 0 <= u < W and 0 <= v < H and labels[v, u] == largest_label]
    return final_uv

def process_frame_improved(bin_path, args):
    frame = bin_path.stem
    img_path = Path(args.image_dir) / f"{frame}.png"
    calib_path = Path(args.calib_dir) / f"{frame}.txt"
    
    if not (img_path.exists() and calib_path.exists()):
        print(f"[WARN] missing assets for {frame}")
        return

    xyz = load_bin(bin_path)
    xyz = xyz[np.abs(xyz[:, 1]) < 8.0]  # Keep your original lateral crop
    pcd = pc_to_o3d(xyz).voxel_down_sample(0.1)

    # Keep your original RANSAC approach
    plane, refined_idx, all_inliers = ransac_ground(pcd, args.dist, args.iters)
    road_pts = np.asarray(pcd.points)[refined_idx]

    # Keep your original local flatness filter
    road_pts = filter_points_by_local_smoothness(road_pts)
    
    # NEW: Balanced sidewalk removal (preserves both lanes)
    road_pts = remove_sidewalks_balanced(road_pts, road_width=7.0, sidewalk_gap=1.5)
    
    # NEW: Remove small isolated regions (less aggressive)
    road_pts = remove_isolated_regions(road_pts, min_cluster_size=20, max_clusters=3)
    
    # Apply tighter lateral crop at the end
    road_pts = road_pts[np.abs(road_pts[:, 1]) < 6]

    # Keep your original projection and visualization
    proj = parse_calib(calib_path)
    img = cv2.imread(str(img_path))

    # Red: all RANSAC inliers
    all_uv = project(np.asarray(pcd.points)[all_inliers], proj)
    for u, v in all_uv:
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.circle(img, (u, v), 1, (0, 0, 255), -1)

    # Blue: filtered road points
    uv = project(road_pts, proj)
    final_uv = filter_largest_connected_component(uv, img.shape)
    for u, v in final_uv:
        cv2.circle(img, (u, v), 2, (255, 0, 0), -1)

    print(f"Processed {frame}: {len(road_pts)} refined road pts, {len(all_inliers)} raw inliers")
    cv2.imshow('Road Projection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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