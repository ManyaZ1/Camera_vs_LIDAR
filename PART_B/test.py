#!/usr/bin/env python3
"""
KITTI LiDAR → Road Mask (improved)
- fixed signed-distance filter to drop sidewalk
- slight relax on smoothness std
Paths remain **unchanged** as per user request.
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
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    return pc

def ransac_ground(pcd, dist, iters):
    plane, inl = pcd.segment_plane(distance_threshold=dist, ransac_n=3, num_iterations=iters)
    refined = refine_inliers_by_distance(np.asarray(pcd.points), plane, inl)
    return plane, list(refined)

def refine_inliers_by_distance(points, plane, idx):
    """Κρατά σημεία πολύ κοντά και *κάτω* από το επίπεδο.
    Signed_dist <= 0 → στο ίδιο ύψος ή χαμηλότερα (δρόμος).
    Επιτρέπουμε 8 cm κάτω / 2 cm πάνω για ανοχή.
    """
    a, b, c, d = plane
    normal = np.array([a, b, c])
    denom = np.linalg.norm(normal)
    signed = (points @ normal + d) / denom
    keep = (signed >= -0.08) & (signed <= 0.02) & (signed <= 0)
    return [i for i in idx if keep[i]]

def filter_points_by_local_smoothness(pts, radius=0.25, height_thresh=0.06):
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
    pcd = pc_to_o3d(xyz).voxel_down_sample(0.1)
    plane, inl = ransac_ground(pcd, args.dist, args.iters)
    road_pts = np.asarray(pcd.points)[inl]
    road_pts = filter_points_by_local_smoothness(road_pts)
    road_pts = road_pts[np.abs(road_pts[:, 1]) < 4]  # lateral crop

    proj = parse_calib(calib_path)
    img = cv2.imread(str(img_path))
    for u, v in project(road_pts, proj):
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.circle(img, (u, v), 2, (255, 0, 0), -1)
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
