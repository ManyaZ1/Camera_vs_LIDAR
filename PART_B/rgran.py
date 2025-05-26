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
def estimate_normals(pts, radius=0.5):
    pcd = pc_to_o3d(pts)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    normals = np.asarray(pcd.normals)
    return normals

def is_valid_plane_normal(normal, threshold_angle_deg=20):
    """
    Accept if normal is within `threshold_angle_deg` of vertical.
    """
    z_axis = np.array([0, 0, 1])
    cos_theta = np.dot(normal, z_axis) / (np.linalg.norm(normal) * np.linalg.norm(z_axis))
    angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    return angle_deg < threshold_angle_deg
def segment_largest_smooth_region(pts: np.ndarray,
                                   radius: float = 0.5,
                                   angle_thresh_deg: float = 15,
                                   eps: float = 0.5,
                                   min_samples: int = 20) -> np.ndarray:
    '''
    Segments largest contiguous smooth surface using normals + DBSCAN.
    '''
    pcd = pc_to_o3d(pts)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    normals = np.asarray(pcd.normals)

    # Only keep points whose normals are near-horizontal (Z axis)
    z_axis = np.array([0, 0, 1])
    cos_angles = np.dot(normals, z_axis)
    angle_thresh = np.cos(np.radians(angle_thresh_deg))
    flat_mask = cos_angles > angle_thresh

    flat_pts = pts[flat_mask]
    if len(flat_pts) < min_samples:
        return np.zeros(len(pts), dtype=bool)

    # Cluster the flat points using DBSCAN
    labels = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(flat_pts))\
        .cluster_dbscan(eps=eps, min_points=min_samples, print_progress=False)

    labels = np.array(labels)
    if labels.max() < 0:
        return np.zeros(len(pts), dtype=bool)

    # Get largest cluster
    largest = np.argmax(np.bincount(labels[labels >= 0]))
    cluster_mask = (labels == largest)

    # Map back to original point indices
    final_mask = np.zeros(len(pts), dtype=bool)
    final_mask[np.where(flat_mask)[0][cluster_mask]] = True
    return final_mask


def refine_inliers_by_distance(points, plane, idx):
    '''Refine inliers by signed distance to the plane'''
    a, b, c, d = plane
    normal = np.array([a, b, c])
    denom = np.linalg.norm(normal)
    signed = (points @ normal + d) / denom
    keep = (signed >= -0.15) & (signed <= 0.07) & (signed <= 0)
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

def project(pts, P):
    h = np.hstack([pts, np.ones((len(pts), 1))])
    uvw = (P @ h.T).T
    uv = (uvw[:, :2] / uvw[:, 2:3]).astype(int)
    return uv

# ────────────────────── Main processing ────────────────────── #

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
    mask = segment_largest_smooth_region(np.asarray(pcd.points))
    road_pts = np.asarray(pcd.points)[mask]



    # Filter points based on local smoothness
    road_pts = filter_points_by_local_smoothness(road_pts)
    road_pts = road_pts[np.abs(road_pts[:, 1]) < 6]

    proj = parse_calib(calib_path)
    img = cv2.imread(str(img_path))

    # # Red: all inliers
    # all_uv = project(np.asarray(pcd.points)[all_inliers], proj)
    # for u, v in all_uv:
    #     if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
    #         cv2.circle(img, (u, v), 1, (0, 0, 255), -1)

    # Blue: final refined road
    for u, v in project(road_pts, proj):
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.circle(img, (u, v), 2, (255, 0, 0), -1)
    #print(f"Processed {frame}: {len(road_pts)} road points, {len(all_inliers)} total inliers")
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
            process_frame(f, a)
        else:
            print(f"missing {f}")
