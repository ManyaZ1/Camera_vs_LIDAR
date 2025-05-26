from pathlib import Path
import sys
import argparse
import numpy as np
import open3d as o3d
import cv2
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from numpy.typing import NDArray
import numpy as np
# ────────────────────── Config ────────────────────── #


def get_args():
    p = argparse.ArgumentParser("KITTI Velodyne viewer + road extraction (v2)")
    p.add_argument("--velodyne_dir", default="C:/Users/USER/Documents/_CAMERA_LIDAR/data_road_velodyne/training/velodyne")
    p.add_argument("--index", default="um_000000")
    p.add_argument("--dist", type=float, default=0.15, help="RANSAC distance threshold (m)")
    p.add_argument("--iters", type=int, default=1000, help="RANSAC iterations")
    p.add_argument("--calib_dir", default="C:/Users/USER/Documents/_CAMERA_LIDAR/data_road/training/calib")
    p.add_argument("--image_dir", default="C:/Users/USER/Documents/_CAMERA_LIDAR/data_road/training/image_2")
    # new parameters (kept optional)
    p.add_argument("--voxel", type=float, default=0.1, help="Voxel size for down‑sampling (m)")
    p.add_argument("--radius", type=float, default=0.5, help="Neighbourhood radius for PCA (m)")
    p.add_argument("--curv_thresh", type=float, default=0.04, help="Curvature threshold for edge detection")
    p.add_argument("--normal_thresh", type=float, default=0.3, help="Normal angle change for edge detection (rad)")
    return p.parse_args()


# ────────────────────── Pre‑processing ────────────────────── #


def load_bin(path: Path) -> NDArray[np.float32]:
    """Read KITTI *.bin* and apply coarse FOV gating."""
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    mask = (
        (pts[:, 0] > 0) & (pts[:, 0] < 60) &  # forward 60 m
        (np.abs(pts[:, 1]) < 20) &           # ±20 m lateral
        (pts[:, 2] > -3) & (pts[:, 2] < 2)   # height gate
    )
    return pts[mask]


def statistical_outlier_removal(xyz: NDArray[np.float32], nb: int = 16, z: float = 2.0):
    """Remove statistical outliers using Open3D wrapper."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=z)
    return xyz[ind]


def organise_into_grid(xyz: NDArray[np.float32], voxel: float):
    """Voxel down‑sample to regularise density; returns same shape points."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    ds = pcd.voxel_down_sample(voxel)
    return np.asarray(ds.points)


# ────────────────────── Ground segmentation ────────────────────── #


def ransac_plane(pcd: o3d.geometry.PointCloud, dist: float, iters: int):
    plane, inliers = pcd.segment_plane(distance_threshold=dist, ransac_n=3, num_iterations=iters)
    return np.array(plane), inliers


def refine_piecewise(xyz: NDArray[np.float32], plane: NDArray[np.float64], slab: float = 2.0):
    """Split forward x‑axis into slabs, fit plane to each to capture curvature."""
    if len(xyz) == 0:
        return np.empty((0, 4))
    x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
    planes = []
    for start in np.arange(x_min, x_max, slab):
        mask = (xyz[:, 0] >= start) & (xyz[:, 0] < start + slab)
        pts = xyz[mask]
        if len(pts) < 50:
            continue
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        try:
            pl, _ = pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=300)
            planes.append(pl)
        except RuntimeError:
            pass  # skip ill‑conditioned slabs
    if planes:
        return np.vstack(planes)
    return np.empty((0, 4))


# ────────────────────── Feature extraction ────────────────────── #


def estimate_normals(pcd: o3d.geometry.PointCloud, radius: float):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    pcd.normalize_normals()
    return pcd


def local_pca_features(xyz: NDArray[np.float32], radius: float):
    """Return curvature λ3/(λ1+λ2+λ3) and normal vectors via PCA."""
    tree = cKDTree(xyz)
    curvature = np.zeros(len(xyz))
    normals = np.zeros((len(xyz), 3))
    for i, p in enumerate(xyz):
        ind = tree.query_ball_point(p, radius)
        if len(ind) < 5:
            curvature[i] = 1.0  # treat as noisy
            normals[i] = np.array([0, 0, 1])
            continue
        nb = xyz[ind]
        cov = np.cov(nb.T)
        w, v = np.linalg.eigh(cov)
        order = w.argsort()[::-1]
        w, v = w[order], v[:, order]
        curvature[i] = w[2] / ( w.sum() + 1e-12) 
        normals[i] = v[:, 2]  # smallest eigenvector → surface normal
    return curvature, normals


def extract_edge_candidates(xyz: NDArray[np.float32], curvature: NDArray[np.float64], normals: NDArray[np.float32], curv_thresh: float, normal_thresh: float):
    curv_mask = curvature > curv_thresh
    # abrupt change in normals (angle > threshold rad)
    tree = cKDTree(xyz)
    norm_mask = np.zeros(len(xyz), dtype=bool)
    for i, n in enumerate(normals):
        ind = tree.query_ball_point(xyz[i], 0.3)
        if len(ind) < 5:
            continue
        angles = np.arccos(np.clip(normals[ind] @ n, -1.0, 1.0))
        if np.max(angles) > normal_thresh:
            norm_mask[i] = True
    return xyz[curv_mask | norm_mask]


# ────────────────────── Road modelling ────────────────────── #


def poly_fit_edge(edge_pts: NDArray[np.float32], deg: int = 3):
    """Fit polynomial y = f(x) to one side (assumes monotonically increasing x)."""
    if len(edge_pts) < deg + 1:
        return np.zeros(deg + 1)
    x = edge_pts[:, 0]
    y = edge_pts[:, 1]
    return np.polyfit(x, y, deg)


def fit_lane_boundaries(edge_pts: NDArray[np.float32]):
    left = edge_pts[edge_pts[:, 1] > 0]  # KITTI coord: +y left
    right = edge_pts[edge_pts[:, 1] < 0]
    left_poly = poly_fit_edge(left) if len(left) else None
    right_poly = poly_fit_edge(right) if len(right) else None
    return left_poly, right_poly


# ────────────────────── Post‑processing ────────────────────── #


def merge_and_filter_edges(edge_pts: NDArray[np.float32]):
    if len(edge_pts) == 0:
        return edge_pts
    clustering = DBSCAN(eps=0.3, min_samples=10).fit(edge_pts[:, :2])
    labels = clustering.labels_
    good = labels != -1  # remove noise
    return edge_pts[good]


# ────────────────────── Utility (projection helpers) ────────────────────── #


def pc_to_o3d(xyz):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    return pc


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


# ────────────────────── Frame processing ────────────────────── #


def process_frame(bin_path: Path, args):
    frame = bin_path.stem
    img_path = Path(args.image_dir) / f"{frame}.png"
    calib_path = Path(args.calib_dir) / f"{frame}.txt"
    if not (img_path.exists() and calib_path.exists()):
        print(f"[WARN] missing assets for {frame}")
        return

    # 1. Load & pre‑process
    xyz = load_bin(bin_path)
    xyz = statistical_outlier_removal(xyz)
    xyz = organise_into_grid(xyz, args.voxel)

    # 2. Ground segmentation
    pcd = pc_to_o3d(xyz)
    plane, inliers = ransac_plane(pcd, args.dist, args.iters)
    ground_pts = xyz[inliers]
    slab_planes = refine_piecewise(ground_pts, plane)  # not used directly but could inform curvature

    # 3. Feature extraction
    curvature, normals = local_pca_features(ground_pts, radius=args.radius)
    edge_candidates = extract_edge_candidates(ground_pts, curvature, normals, args.curv_thresh, args.normal_thresh)

    # 4. Road modelling
    edge_candidates = merge_and_filter_edges(edge_candidates)
    left_poly, right_poly = fit_lane_boundaries(edge_candidates)

    # 5. Visualisation on camera image
    proj = parse_calib(calib_path)
    img = cv2.imread(str(img_path))

    # All ground inliers = red
    for u, v in project(ground_pts, proj):
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            img[v, u] = (0, 0, 255)

    # Edge candidates = blue
    for u, v in project(edge_candidates, proj):
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            img[v, u] = (255, 0, 0)

    # Draw fitted lane boundaries (green)
    if left_poly is not None:
        xs = np.linspace(0, 40, 200)
        ys = np.polyval(left_poly, xs)
        lane = np.vstack([xs, ys, np.zeros_like(xs)]).T
        for u, v in project(lane, proj):
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                img[v, u] = (0, 255, 0)
    if right_poly is not None:
        xs = np.linspace(0, 40, 200)
        ys = np.polyval(right_poly, xs)
        lane = np.vstack([xs, ys, np.zeros_like(xs)]).T
        for u, v in project(lane, proj):
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                img[v, u] = (0, 255, 0)

    cv2.imshow("Road Detection v2", img)
    cv2.waitKey(0)  # minimal delay to allow automatic playback
    cv2.destroyAllWindows()

# ────────────────────── Entry point ────────────────────── #


if __name__ == "__main__":
    a = get_args()
    v_dir = Path(a.velodyne_dir)
    if not v_dir.exists():
        sys.exit(f"[ERR] Velodyne dir not found: {v_dir}")

    files = (
        sorted(v_dir.glob("*.bin")) if a.index.lower() == "all" else [v_dir / f"{a.index}.bin"]
    )
    for f in files:
        if f.exists():
            process_frame(f, a)
        else:
            print(f"[WARN] missing {f}")