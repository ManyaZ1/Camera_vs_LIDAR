
# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
Show a KITTI Velodyne frame, add a virtual wall, visualise the point-cloud,
and project *all* points (original + wall) onto the corresponding left-camera
image.

Usage example
-------------
python view_and_project.py --index um_000055 --wall_x 12
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import cv2
#--index=um_000090 --velodyne_dir=C:/Users/USER/Documents/_CAMERA_LIDAR/data_road_velodyne/testing/velodyne --calib_dir=C:/Users/USER/Documents/_CAMERA_LIDAR/data_road/testing/calib --image_dir=C:/Users/USER/Documents/_CAMERA_LIDAR/data_road/testing/image_2 --output_dir=C:/Users/USER/Documents/GitHub/Camera_vs_LIDAR/PART_B/fakebin
'''--index=um_000017 --velodyne_dir=C:/Users/USER/Documents/_CAMERA_LIDAR/data_road_velodyne/testing/velodyne --calib_dir=C:/Users/USER/Documents/_CAMERA_LIDAR/data_road/testing/calib --image_dir=C:/Users/USER/Documents/_CAMERA_LIDAR/data_road/testing/image_2 --output_dir=C:/Users/USER/Documents/GitHub/Camera_vs_LIDAR/PART_B/fakebin'''
# ──────────────── command-line arguments ──────────────── #

def get_args():
    p = argparse.ArgumentParser("Velodyne cloud viewer + projection")
    # p.add_argument("--velodyne_dir",
    #                default="C:/Users/Mania/Documents/KITTI/data_road_velodyne/training/velodyne")
    # p.add_argument("--calib_dir",
    #                default="C:/Users/Mania/Documents/KITTI/data_road/training/calib")
    # p.add_argument("--image_dir",
    #                default="C:/Users/Mania/Documents/KITTI/data_road/training/image_2")
    p.add_argument("--index", default="umm_000090")
    p.add_argument("--velodyne_dir", default="C:/Users/USER/Documents/_CAMERA_LIDAR/data_road_velodyne/training/velodyne")
    #p.add_argument("--index", default="all")
    p.add_argument("--calib_dir", default="C:/Users/USER/Documents/_CAMERA_LIDAR/data_road/training/calib")
    p.add_argument("--image_dir",  default="C:/Users/USER/Documents/_CAMERA_LIDAR/data_road/training/image_2")
    p.add_argument("--dist", type=float, default=0.15)
    # virtual wall parameters
    p.add_argument("--wall_x",      type=float, default=8.0)
    p.add_argument("--wall_width",  type=float, default=10.0)
    p.add_argument("--wall_height", type=float, default=2.0)
    p.add_argument("--wall_step",   type=float, default=0.1)
    p.add_argument("--wall_depth", type=float, default=0.5, help="Thickness of the virtual wall")
    p.add_argument("--output_dir", default="C:/Users/USER/Documents/GitHub/Camera_vs_LIDAR/PART_B/fakebin")
    return p.parse_args()


# ────────────── basic data helpers ────────────── #

def load_bin(path: Path) -> np.ndarray:
    """Velodyne *.bin → N×3 xyz (float32)."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def make_wall(x_front: float, width: float, height: float, step: float) -> np.ndarray:
    """Regular grid of points on plane x = x_front."""
    
    y = np.arange(-width / 2,  width / 2 + step, step)
    z = np.arange(-height / 2, height / 2 + step, step)
    yy, zz = np.meshgrid(y, z)
    xx = np.full_like(yy, x_front)
    return np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)
def make_wall_super_dense(x_front: float, width: float, height: float, step: float, depth: float = 0.5) -> np.ndarray:
    """
    Generates a volumetric wall starting at x_front, with width, height, and depth.
    """
    x = np.arange(x_front, x_front + depth + step, step)
    y = np.arange(-width / 2,  width / 2 + step, step)
    z = np.arange(-height / 2, height / 2 + step, step)

    xx, yy, zz = np.meshgrid(x, y, z)
    wall_points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)
    return wall_points

# ──────────── calibration & projection ──────────── #

def parse_calib(txt_path: Path) -> np.ndarray:
    """
    Return 4×4 projection matrix P_cam0 ← Velodyne

    KITTI file provides:
      P2         (3×4)  — left-camera projection
      R0_rect    (3×3)  — rectification
      Tr_velo_to_cam (3×4) — Velodyne → cam0
    """
    data = {}
    with open(txt_path) as f:
        for line in f:
            if ':' in line:
                k, v = line.strip().split(':', 1)
                data[k] = np.fromstring(v, sep=' ')
    P2 = data['P2'].reshape(3, 4)
    R0 = data['R0_rect'].reshape(3, 3)
    Tr = data['Tr_velo_to_cam'].reshape(3, 4)

    T_velo_cam = np.eye(4); T_velo_cam[:3, :4] = Tr
    R_rect     = np.eye(4); R_rect[:3, :3]     = R0
    P_rect0    = np.eye(4); P_rect0[:3, :4]    = P2

    return P_rect0 @ R_rect @ T_velo_cam      # 4×4


def project(pts: np.ndarray, P: np.ndarray, img_shape) -> np.ndarray:
    """
    3-D points → 2-D pixel coords, discard those behind camera or outside image.
    Returns N×2 int array.
    """
    if len(pts) == 0:
        return np.empty((0, 2), np.int32)

    ones = np.ones((len(pts), 1), dtype=pts.dtype)
    uvw  = (P @ np.hstack([pts, ones]).T).T
    z    = uvw[:, 2]
    valid = z > 0
    uv = np.zeros((len(pts), 2), np.int32)
    uv[valid] = (uvw[valid, :2] / z[valid, None]).astype(np.int32)

    h, w = img_shape[:2]
    in_img = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv[valid & in_img]


# ─────────────── visualisation helpers ─────────────── #

def np_to_pcd(points: np.ndarray, color=(1.0, 1.0, 1.0)) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc.paint_uniform_color(color)
    return pc


# ───────────────────────── main ───────────────────────── #

def main():
    a = get_args()
    bin_path   = Path(a.velodyne_dir) / f"{a.index}.bin"
    calib_path = Path(a.calib_dir)    / f"{a.index}.txt"
    img_path   = Path(a.image_dir)    / f"{a.index}.png"
    output_path = Path(a.output_dir) / f"f{a.index}.bin"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not (bin_path.exists() and calib_path.exists() and img_path.exists()):
        raise FileNotFoundError("Some KITTI files are missing")

    cloud  = load_bin(bin_path)
    cloud = cloud[cloud[:, 0] > 0]  # Keep only points in front (x > 0)
    #wall   = make_wall(a.wall_x, a.wall_width, a.wall_height, a.wall_step)
    wall = make_wall(a.wall_x, a.wall_width, a.wall_height, a.wall_step)

    merged = np.vstack([cloud, wall])

    # ── Save merged point cloud ── #
    merged_with_reflectance = np.hstack([merged, np.ones((merged.shape[0], 1), dtype=np.float32)])

    
    merged_with_reflectance.astype(np.float32).tofile(output_path)
    print(f"Saved merged point cloud to {output_path}")
    # ── Open3D view ── #
    # o3d.visualization.draw_geometries(
    #     [np_to_pcd(cloud, (0.9, 0.9, 0.9)), np_to_pcd(wall, (1.0, 0.0, 0.0))],
    #     window_name=f"{a.index} | white=original  red=wall"
    # )
    cloud_pcd = np_to_pcd(cloud, (0.9, 0.9, 0.9))  # light grey
    wall_pcd  = np_to_pcd(wall,  (1.0, 0.0, 0.0))  # red

    o3d.visualization.draw_geometries(
        [cloud_pcd, wall_pcd],
        window_name=f"{a.index} | white=original  red=wall"
    )

    # ── Projection to image ── #
    P = parse_calib(calib_path)
    img = cv2.imread(str(img_path))

    # # to reduce over-plotting, you may subsample; keep full density here
    # uv = project(merged, P, img.shape)

    # # draw tiny circles (blue wall on red background to distinguish)
    # for (u, v) in uv:
    #     img[v, u] = (255, 0, 0)      # BGR (blue) for visibility
    uv_cloud = project(cloud, P, img.shape)
    uv_wall  = project(wall,  P, img.shape)

    # # Draw original cloud points (small grey dots)
    # for (u, v) in uv_cloud:
    #     cv2.circle(img, (u, v), radius=1, color=(160, 160, 160), thickness=-1)

    # Draw wall points (bigger red dots)
    for (u, v) in uv_wall:
        cv2.circle(img, (u, v), radius=4, color=(0, 0, 255), thickness=-1)
    
    
    cv2.imshow("Projection of full cloud (merged)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
