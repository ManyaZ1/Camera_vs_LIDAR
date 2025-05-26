#!/usr/bin/env python3
"""
Quick‑start KITTI Velodyne viewer + road extraction (RANSAC)
Now with projection of road/non-road points over the RGB image.
Improved filtering using plane proximity and lateral cropping.
"""

from pathlib import Path
import sys
import argparse
import numpy as np
import open3d as o3d
import cv2

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

def load_bin(bin_path: Path):
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    xyz = pts[:, :3]
    mask = (
        (xyz[:, 0] > 0.0) & (xyz[:, 0] < 40.0) &
        (np.abs(xyz[:, 1]) < 15.0) &
        (xyz[:, 2] > -2.0) & (xyz[:, 2] < 0.5)
    )
    return xyz[mask]

def pc_to_o3d(xyz: np.ndarray):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    return pc

def ransac_ground(pcd: o3d.geometry.PointCloud, dist: float, iters: int):
    plane_model, inliers = pcd.segment_plane(distance_threshold=dist, ransac_n=3, num_iterations=iters)
    refined = refine_inliers_by_distance(np.asarray(pcd.points), plane_model, inliers)
    return list(refined), plane_model

def refine_inliers_by_distance(points: np.ndarray, plane_model, inlier_indices):
    a, b, c, d = plane_model
    denom = np.sqrt(a ** 2 + b ** 2 + c ** 2)
    dists = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / denom
    refined = [i for i in inlier_indices if dists[i] < 0.08]  # πιο ευέλικτο φίλτρο
    return refined

def parse_calib_txt(txt_path):
    calib = {}
    with open(txt_path, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            key, value = line.strip().split(':', 1)
            calib[key] = np.array([float(x) for x in value.strip().split()])

    P2 = calib['P2'].reshape(3, 4)
    R0_rect = calib['R0_rect'].reshape(3, 3)
    Tr_velo_to_cam = calib['Tr_velo_to_cam'].reshape(3, 4)

    Tr = np.eye(4); Tr[:3, :4] = Tr_velo_to_cam
    R = np.eye(4); R[:3, :3] = R0_rect
    P = np.eye(4); P[:3, :4] = P2

    return P @ R @ Tr

def project_lidar_to_img(points_3d, proj_mat):
    n = points_3d.shape[0]
    homog = np.hstack([points_3d, np.ones((n, 1))])
    proj = (proj_mat @ homog.T).T
    proj[:, 0] /= proj[:, 2]
    proj[:, 1] /= proj[:, 2]
    return proj[:, :2]

# ────────────────────── Main ────────────────────── #

def process_file(bin_path: Path, dist: float, iters: int, img_dir: Path, calib_dir: Path):
    frame_id = bin_path.stem
    image_path = img_dir / f"{frame_id}.png"
    calib_path = calib_dir / f"{frame_id}.txt"

    xyz = load_bin(bin_path)
    pcd = pc_to_o3d(xyz).voxel_down_sample(voxel_size=0.1)

    print(f"[INFO] Showing raw cloud for {bin_path.name}. Close window to continue…")
    #o3d.visualization.draw_geometries([pcd])

    inliers, _ = ransac_ground(pcd, dist, iters)
    road = pcd.select_by_index(inliers)

    road_points = np.asarray(road.points)
    mask = np.abs(road_points[:, 1]) < 4.0
    road = road.select_by_index(np.where(mask)[0])

    non_road = pcd.select_by_index(inliers, invert=True)

    road.paint_uniform_color([0.0, 0.0, 1.0])
    non_road.paint_uniform_color([1.0, 0.0, 0.0])
    print(f"[INFO] Road points: {len(road.points)} / {len(pcd.points)}")
    #o3d.visualization.draw_geometries([road + non_road])

    if image_path.exists() and calib_path.exists():
        proj_matrix = parse_calib_txt(calib_path)
        img = cv2.imread(str(image_path))
        road_uv = project_lidar_to_img(np.asarray(road.points), proj_matrix)
        nonroad_uv = project_lidar_to_img(np.asarray(non_road.points), proj_matrix)

        for u, v in road_uv:
            u, v = int(round(u)), int(round(v))
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (u, v), 2, (255, 0, 0), -1)
        for u, v in nonroad_uv:
            u, v = int(round(u)), int(round(v))
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (u, v), 2, (0, 0, 255), -1)

        cv2.imshow("Road Projection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    args = get_args()
    velodyne_dir = Path(args.velodyne_dir)
    if not velodyne_dir.is_dir():
        sys.exit(f"Velodyne dir {velodyne_dir} not found")

    files = sorted(velodyne_dir.glob("*.bin")) if args.index.lower() == "all" else [velodyne_dir / f"{args.index}.bin"]

    for f in files:
        if not f.exists():
            print(f"[WARN] {f} missing – skip")
            continue
        process_file(f, dist=args.dist, iters=args.iters,
                     img_dir=Path(args.image_dir), calib_dir=Path(args.calib_dir))
