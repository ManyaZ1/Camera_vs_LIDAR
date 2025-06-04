# Recommended Improvements
# 1. Use Piecewise or Multiple Plane Fitting

'''#Piecewise RANSAC: Instead of a single plane, fit multiple planes or use region-growing after RANSAC 
# to separate road from sidewalks and grass. '''

# 2. Enhance Local Smoothness Criteria
# 3D Smoothness: Consider both XY and Z (height) variance in the neighborhood, not just XY flatness.

# Normal Vector Consistency: Check if the local normal vectors are consistent
#  with the expected road normal (typically, near-horizontal).

# 5. Post-processing
# Morphological Operations: Use connected component analysis to keep the 
# largest contiguous road region, discarding isolated patches on sidewalks or grass.

#!/usr/bin/env python3
"""
KITTI LiDAR → Road Mask (robust + curved support)
- Red overlay for all RANSAC inliers

Keep points within ±8 m from the center.

Refine inliers by signed distance to the plane.

Run RANSAC to find the ground plane.

Filter points by local smoothness (retain only locally flat points in XY).

Filter detected road points to keep only those within ±6 m of the centerline.

"""

from pathlib import Path
import sys
import argparse
import numpy as np
import open3d as o3d
import cv2
from scipy.spatial import cKDTree

# ────────────────────── Config ────────────────────── #

def get_args():
    p = argparse.ArgumentParser("KITTI Velodyne viewer + road RANSAC")
    p.add_argument("--velodyne_dir", default="C:/Users/USER/Documents/_CAMERA_LIDAR/data_road_velodyne/training/velodyne")
    p.add_argument("--index", default="um_000000")
    p.add_argument("--dist", type=float, default=0.15)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--calib_dir", default="C:/Users/USER/Documents/_CAMERA_LIDAR/data_road/training/calib")
    p.add_argument("--image_dir",  default="C:/Users/USER/Documents/_CAMERA_LIDAR/data_road/training/image_2")
    return p.parse_args()

# ────────────────────── Helpers ────────────────────── #

def load_bin(path: Path) -> np.ndarray:
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    mask = (
        (pts[:, 0] > 0) & (pts[:, 0] < 40) &
        (np.abs(pts[:, 1]) < 15) &
        (pts[:, 2] > -2) & (pts[:, 2] < 0.5)
    )
    return pts[mask]

def pc_to_o3d(xyz):
    '''Convert numpy array to Open3D PointCloud'''
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    return pc

def ransac_ground(pcd, dist, iters):
    '''Run RANSAC to find ground plane in point cloud'''
    plane, inl = pcd.segment_plane(distance_threshold=dist, ransac_n=3, num_iterations=iters)
    refined = refine_inliers_by_distance(np.asarray(pcd.points), plane, inl)
    return plane, list(refined), list(inl)  # return also raw inliers

def refine_inliers_by_distance(points, plane, idx):
    '''Refine inliers by signed distance to the plane'''
    a, b, c, d = plane
    normal = np.array([a, b, c])
    denom = np.linalg.norm(normal)
    signed = (points @ normal + d) / denom
    keep = (signed >= -0.15) & (signed <= 0) #& (signed <= 0.05)
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

def parse_calib(txt: Path):
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

# def project(pts, P):
#     h = np.hstack([pts, np.ones((len(pts), 1))])
#     uvw = (P @ h.T).T
#     uv = (uvw[:, :2] / uvw[:, 2:3]).astype(int)
#     return uv
def project(pts, P):
    h = np.hstack([pts, np.ones((len(pts), 1))])
    uvw = (P @ h.T).T
    z = uvw[:, 2]
    valid = z > 0  # keep only points in front of camera
    uv = np.zeros((len(pts), 2), dtype=int)
    uv[valid] = (uvw[valid, :2] / z[valid, np.newaxis]).astype(int)
    return uv[valid]

# ────────────────────── Main processing ────────────────────── #
def filter_largest_connected_component(uv: np.ndarray, img_shape: tuple) -> list:
    """
    Keeps only the (u, v) points that belong to the largest connected component.

    Args:
        uv: np.ndarray of shape (N, 2) with projected (u, v) points.
        img_shape: shape of the image (height, width).

    Returns:
        Filtered list of (u, v) tuples in the largest component.
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

def process_frame_new(bin_path, args):
    frame = bin_path.stem
    img_path = Path(args.image_dir) / f"{frame}.png"
    calib_path = Path(args.calib_dir) / f"{frame}.txt"
    if not (img_path.exists() and calib_path.exists()):
        print(f"[WARN] missing assets for {frame}")
        return

    xyz = load_bin(bin_path)
    xyz = xyz[np.abs(xyz[:, 1]) < 8.0]  # lateral crop
    pcd = pc_to_o3d(xyz).voxel_down_sample(0.1)

    # RANSAC
    plane, refined_idx, all_inliers = ransac_ground(pcd, args.dist, args.iters)
    road_pts = np.asarray(pcd.points)[refined_idx]

    # Local flatness
    road_pts = filter_points_by_local_smoothness(road_pts)
    road_pts = road_pts[np.abs(road_pts[:, 1]) < 6]  # tighter lateral crop

    # Load projection
    proj = parse_calib(calib_path)
    img = cv2.imread(str(img_path))

    # Red: all RANSAC inliers
    all_uv = project(np.asarray(pcd.points)[all_inliers], proj)
    for u, v in all_uv:
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.circle(img, (u, v), 1, (0, 0, 255), -1)

    # Blue: only largest road blob
    uv = project(road_pts, proj)
    final_uv = filter_largest_connected_component(uv, img.shape)
    for u, v in final_uv:
        cv2.circle(img, (u, v), 2, (255, 0, 0), -1)

    print(f"Processed {frame}: {len(road_pts)} refined road pts, {len(all_inliers)} raw inliers")
    cv2.imshow('Road Projection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_frame(bin_path, args):
    frame = bin_path.stem
    img_path = Path(args.image_dir) / f"{frame}.png"
    calib_path = Path(args.calib_dir) / f"{frame}.txt"
    if not (img_path.exists() and calib_path.exists()):
        print(f"[WARN] missing assets for {frame}")
        return

    xyz = load_bin(bin_path)
    # Keep points within a reasonable range
    xyz = xyz[np.abs(xyz[:, 1]) < 8.0]  # κρατάω μόνο ±8 m από το κέντρο
    pcd = pc_to_o3d(xyz).voxel_down_sample(0.1)
    # Run RANSAC to find ground plane
    plane, refined_idx, all_inliers = ransac_ground(pcd, args.dist, args.iters)

    road_pts = np.asarray(pcd.points)[refined_idx]

    # Filter points based on local smoothness
    road_pts = filter_points_by_local_smoothness(road_pts)
    road_pts = road_pts[np.abs(road_pts[:, 1]) < 6]

    proj = parse_calib(calib_path)
    img = cv2.imread(str(img_path))

    # Red: all inliers
    all_uv = project(np.asarray(pcd.points)[all_inliers], proj)
    for u, v in all_uv:
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.circle(img, (u, v), 1, (0, 0, 255), -1)

    # Project refined road points
    uv = project(road_pts, proj)

    # Create a binary mask
    mask_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for u, v in uv:
        if 0 <= u < mask_img.shape[1] and 0 <= v < mask_img.shape[0]:
            mask_img[v, u] = 255

    # Morphological smoothing and largest component selection
    kernel = np.ones((5, 5), np.uint8)
    mask_img = cv2.dilate(mask_img, np.ones((7, 7), np.uint8), iterations=2)
    mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_img)

    # Compute vertical center of each component
    valid_labels = []
    for label in range(1, nlabels):
        y_top = stats[label, cv2.CC_STAT_TOP]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        y_center = y_top + h // 2
        if y_center > img.shape[0] * 0.4:  # keep only components near bottom half
            valid_labels.append(label)

    # Build final mask
    final_uv = []
    for (u, v) in uv:
        if 0 <= u < labels.shape[1] and 0 <= v < labels.shape[0]:
            if labels[v, u] in valid_labels:
                final_uv.append((u, v))

    # Draw only final filtered road points
    for u, v in final_uv:
        cv2.circle(img, (u, v), 2, (255, 0, 0), -1)
      
    
    print(f"Processed {frame}: {len(road_pts)} road points, {len(all_inliers)} total inliers")
    
    # show
    cv2.imshow('Road Projection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ────────────────────── Entry point ────────────────────── #

if __name__ == '__main__':
    a = get_args()
    v_dir = Path(a.velodyne_dir)
    files = sorted(v_dir.glob('*.bin')) if a.index.lower() == 'all' else [v_dir / f"{a.index}.bin"]
    for f in files:
        if f.exists():
            process_frame_new(f, a)
        else:
            print(f"missing {f}")
