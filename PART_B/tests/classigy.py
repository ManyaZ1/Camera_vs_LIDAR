import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import cv2
from pathlib import Path
import argparse
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
import torch
import torchvision.transforms as transforms
from torchvision import models
import requests
from PIL import Image
import io

LOCAL_HEIGHT_TRESHOLD = 0.1 # adaptive_height_filtering
HEIGHT_VARIATION_THRESHOLD = 0.1  # road_continuity_filter

# Classification setup
IMAGENET_CLASSES = [
    'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 'dog', 'cat',
    # Add more relevant classes as needed
]

def setup_classifier():
    """Setup pre-trained neural network for object classification"""
    try:
        # Load pre-trained ResNet50
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.eval()
        
        # Image preprocessing pipeline
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Load ImageNet class labels
        LABELS_URL = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
        try:
            response = requests.get(LABELS_URL)
            labels = response.text.strip().split('\n')
        except:
            # Fallback labels if download fails
            labels = [f'class_{i}' for i in range(1000)]
        
        return model, preprocess, labels
    except Exception as e:
        print(f"Warning: Could not setup classifier: {e}")
        return None, None, None

def get_args():
    p = argparse.ArgumentParser("KITTI Velodyne viewer + road RANSAC")
    p.add_argument("--velodyne_dir", default="C:/Users/Mania/Documents/KITTI/data_road_velodyne/training/velodyne")
    p.add_argument("--index", default="all")
    p.add_argument("--dist", type=float, default=0.15)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--calib_dir", default="C:/Users/Mania/Documents/KITTI/data_road/training/calib")
    p.add_argument("--image_dir",  default="C:/Users/Mania/Documents/KITTI/data_road/training/image_2")
    
    return p.parse_args()

def extract_object_roi(img, bbox, padding=10):
    """Extract region of interest from image for classification"""
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    
    # Add padding and ensure bounds
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    roi = img[y1:y2, x1:x2]
    return roi

def classify_object(roi, model, preprocess, labels):
    """Classify object in ROI using pre-trained model"""
    if model is None or roi.size == 0:
        return "unknown", 0.0
    
    try:
        # Convert BGR to RGB
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(roi_rgb)
        
        # Preprocess and predict
        input_tensor = preprocess(pil_image)
        input_batch = input_tensor.unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
        # Get top prediction
        top_prob, top_class = torch.topk(probabilities, 1)
        class_name = labels[top_class.item()]
        confidence = top_prob.item()
        
        return class_name, confidence
    except Exception as e:
        print(f"Classification error: {e}")
        return "unknown", 0.0

def compute_3d_bounding_box(points):
    """Compute 3D bounding box from point cloud"""
    if len(points) == 0:
        return None
    
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    
    # Define 8 corners of the bounding box
    corners = np.array([
        [min_coords[0], min_coords[1], min_coords[2]],  # min corner
        [max_coords[0], min_coords[1], min_coords[2]],
        [max_coords[0], max_coords[1], min_coords[2]],
        [min_coords[0], max_coords[1], min_coords[2]],
        [min_coords[0], min_coords[1], max_coords[2]],
        [max_coords[0], min_coords[1], max_coords[2]],
        [max_coords[0], max_coords[1], max_coords[2]],  # max corner
        [min_coords[0], max_coords[1], max_coords[2]]
    ])
    
    return corners

def project_3d_bbox_to_image(bbox_3d, proj):
    """Project 3D bounding box to image coordinates"""
    if bbox_3d is None:
        return None
    
    # Project all 8 corners
    projected = project(bbox_3d, proj)
    
    if len(projected) == 0:
        return None
    
    # Get 2D bounding box from projected corners
    x_coords = projected[:, 0]
    y_coords = projected[:, 1]
    
    x1, y1 = np.min(x_coords), np.min(y_coords)
    x2, y2 = np.max(x_coords), np.max(y_coords)
    
    return (int(x1), int(y1), int(x2), int(y2))

def draw_3d_bbox_wireframe(img, bbox_3d, proj, color=(0, 255, 0), thickness=2):
    """Draw 3D bounding box wireframe on image"""
    if bbox_3d is None:
        return img
    
    projected = project(bbox_3d, proj)
    if len(projected) < 8:
        return img
    
    # Define the 12 edges of a cube
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
    ]
    
    h, w = img.shape[:2]
    
    for start_idx, end_idx in edges:
        if start_idx < len(projected) and end_idx < len(projected):
            pt1 = tuple(projected[start_idx])
            pt2 = tuple(projected[end_idx])
            
            # Check if points are within image bounds
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                cv2.line(img, pt1, pt2, color, thickness)
    
    return img

def detect_and_classify_obstacles(all_points, road_points, img, proj, 
                                model=None, preprocess=None, labels=None,
                                height_threshold=0.5, min_cluster_size=10):
    """Enhanced obstacle detection with bounding boxes and classification"""
    if len(road_points) == 0:
        return [], img
    
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
    
    if len(obstacle_points) == 0:
        return [], img
    
    obstacle_points = np.array(obstacle_points)
    
    # Cluster obstacle points
    clustering = DBSCAN(eps=0.5, min_samples=min_cluster_size)
    labels_cluster = clustering.fit_predict(obstacle_points)
    
    detected_objects = []
    
    # Process each cluster
    for label in set(labels_cluster):
        if label == -1:  # Skip noise
            continue
        
        cluster_points = obstacle_points[labels_cluster == label]
        
        if len(cluster_points) < min_cluster_size:
            continue
        
        # Compute 3D bounding box
        bbox_3d = compute_3d_bounding_box(cluster_points)
        
        # Project to 2D
        bbox_2d = project_3d_bbox_to_image(bbox_3d, proj)
        
        if bbox_2d is None:
            continue
        
        x1, y1, x2, y2 = bbox_2d
        
        # Ensure bounding box is valid and within image bounds
        h, w = img.shape[:2]
        if x2 <= x1 or y2 <= y1 or x1 >= w or y1 >= h or x2 <= 0 or y2 <= 0:
            continue
        
        # Clip to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Extract ROI for classification
        roi = extract_object_roi(img, (x1, y1, x2, y2))
        class_name, confidence = classify_object(roi, model, preprocess, labels)
        
        # Filter for relevant object classes
        relevant_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 
                          'traffic_light', 'stop_sign', 'fire_hydrant']
        
        is_relevant = any(rel_class in class_name.lower() for rel_class in relevant_classes)
        
        detected_objects.append({
            'points': cluster_points,
            'bbox_2d': (x1, y1, x2, y2),
            'bbox_3d': bbox_3d,
            'class': class_name,
            'confidence': confidence,
            'is_relevant': is_relevant or confidence < 0.3  # Include low-confidence as unknown objects
        })
        
        # Draw bounding box
        color = (0, 255, 0) if is_relevant else (0, 255, 255)  # Green for relevant, Yellow for others
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw 3D wireframe
        img = draw_3d_bbox_wireframe(img, bbox_3d, proj, color, 1)
        
        # Add label
        label_text = f"{class_name}: {confidence:.2f}"
        if len(label_text) > 25:  # Truncate long class names
            label_text = class_name[:20] + f"...: {confidence:.2f}"
        
        # Background for text
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
        cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return detected_objects, img

# Keep all your existing functions (pc_to_o3d, multi_plane_ransac, etc.)
def pc_to_o3d(xyz):
    '''Convert numpy array to Open3D PointCloud'''
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    return pc

def multi_plane_ransac(pcd, max_planes=3, dist_thresh=0.1, min_points=100):
    '''Find multiple ground planes using iterative RANSAC'''
    points = np.asarray(pcd.points)
    remaining_indices = set(range(len(points))) 
    planes = []
    
    for _ in range(max_planes):
        if len(remaining_indices) < min_points:
            break
            
        sub_indices = list(remaining_indices)
        sub_pcd = o3d.geometry.PointCloud()
        sub_pcd.points = o3d.utility.Vector3dVector(points[sub_indices])
        
        plane, inliers = sub_pcd.segment_plane( 
            distance_threshold=dist_thresh, 
            ransac_n=3, 
            num_iterations=1000
        )
        
        if len(inliers) < min_points:
            break
        
        original_inliers = [sub_indices[i] for i in inliers]
        planes.append((plane, original_inliers))
        
        remaining_indices -= set(original_inliers)
    
    return planes

def adaptive_height_filtering(points, grid_size=1.0, percentile=10):
    '''Adaptive height filtering based on local ground level'''
    if len(points) == 0:
        return np.array([]).reshape(0, 3)
    
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    x_bins = int((x_max - x_min) / grid_size) + 1
    y_bins = int((y_max - y_min) / grid_size) + 1
    
    x_idx = ((points[:, 0] - x_min) / grid_size).astype(int)
    y_idx = ((points[:, 1] - y_min) / grid_size).astype(int)
    
    x_idx = np.clip(x_idx, 0, x_bins - 1)
    y_idx = np.clip(y_idx, 0, y_bins - 1)
    
    ground_levels = np.full((x_bins, y_bins), np.nan)
    
    for i in range(x_bins):
        for j in range(y_bins):
            mask = (x_idx == i) & (y_idx == j)
            if np.sum(mask) > 0:
                ground_levels[i, j] = np.percentile(points[mask, 2], percentile)
    
    from scipy.interpolate import griddata
    valid_mask = ~np.isnan(ground_levels)
    if np.sum(valid_mask) > 3:
        xi, yi = np.meshgrid(range(x_bins), range(y_bins), indexing='ij')
        valid_points = np.column_stack([xi[valid_mask], yi[valid_mask]])
        valid_values = ground_levels[valid_mask]
        
        all_points = np.column_stack([xi.ravel(), yi.ravel()])
        interpolated = griddata(valid_points, valid_values, all_points, method='linear')
        ground_levels = interpolated.reshape(x_bins, y_bins)
    
    filtered_indices = []
    for idx, (x, y, z) in enumerate(points):
        local_ground = ground_levels[x_idx[idx], y_idx[idx]]
        if not np.isnan(local_ground) and (z - local_ground) < LOCAL_HEIGHT_TRESHOLD:
            filtered_indices.append(idx)
    
    return points[filtered_indices]

def road_continuity_filter(points, search_radius=2.0, min_neighbors=5):
    '''Filter points based on road continuity'''
    if len(points) < min_neighbors:
        return points
    
    tree = cKDTree(points[:, :2])
    valid_indices = []
    
    for i, point in enumerate(points):
        neighbors = tree.query_ball_point(point[:2], search_radius)
        
        if len(neighbors) >= min_neighbors:
            neighbor_points = points[neighbors]
            height_std = np.std(neighbor_points[:, 2])
            if height_std < HEIGHT_VARIATION_THRESHOLD:
                valid_indices.append(i)
    
    return points[valid_indices]

def cluster_road_segments(points, eps=1.0, min_samples=10):
    '''Cluster road points into coherent segments'''
    if len(points) < min_samples:
        return [points]
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(points[:, :2])
    
    clusters = []
    for label in set(labels):
        if label != -1:
            cluster_points = points[labels == label]
            if len(cluster_points) >= min_samples:
                clusters.append(cluster_points)
    
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

def pca_high_variation_mask(points, radius=0.3, z_variance_thresh=0.002):
    """Return a boolean mask of points with high vertical roughness in local patch."""
    kdt = cKDTree(points[:, :2])
    high_variance_mask = np.zeros(len(points), dtype=bool)
    
    for i, p in enumerate(points):
        idx = kdt.query_ball_point(p[:2], radius) 
        if len(idx) < 5:
            continue
        patch = points[idx]
        pca = PCA(n_components=3).fit(patch)
        if pca.explained_variance_[2] > z_variance_thresh:
            high_variance_mask[i] = True
    return high_variance_mask

def process_frame_improved(bin_path, args):
    frame = bin_path.stem
    img_path = Path(args.image_dir) / f"{frame}.png"
    calib_path = Path(args.calib_dir) / f"{frame}.txt"
    
    if not (img_path.exists() and calib_path.exists()):
        print(f"[WARN] missing assets for {frame}")
        return

    # Setup classifier (do this once at startup for better performance)
    model, preprocess, labels = setup_classifier()

    # Load and preprocess
    xyz = load_bin(bin_path)
    xyz = xyz[np.abs(xyz[:, 1]) < 10.0]
    xyz = xyz[xyz[:, 0] > 0]
    xyz = xyz[xyz[:, 2] > -3.0]
    xyz = xyz[xyz[:, 2] < 2.0]
    
    pcd = pc_to_o3d(xyz).voxel_down_sample(0.08)
    
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
    road_points = road_candidates
    
    # Cluster road segments
    road_clusters = cluster_road_segments(road_points, eps=1.5, min_samples=15)
    
    if road_clusters:
        main_road = max(road_clusters, key=len)
        main_road = main_road[np.abs(main_road[:, 1]) < 5.0]

        pca_curb_mask = pca_high_variation_mask(main_road, radius=0.3, z_variance_thresh=0.0001)
        rough_points = main_road[pca_curb_mask]
        main_road = main_road[~pca_curb_mask]
    else:
        main_road = np.array([]).reshape(0, 3)
        rough_points = np.array([]).reshape(0, 3)
    
    # Sidewalk detection logic (keep your existing logic)
    road_center_y = np.median(main_road[:, 1])
    left_rough = rough_points[rough_points[:, 1] < road_center_y]
    right_rough = rough_points[rough_points[:, 1] > road_center_y]
    
    MIN_CANDIDATES = 350
    
    if len(left_rough) > MIN_CANDIDATES:
        left_curb_y = np.percentile(left_rough[:, 1], 95)
    else:
        left_curb_y = -np.inf
        
    if len(right_rough) > MIN_CANDIDATES:
        right_curb_y = np.percentile(right_rough[:, 1], 5)
    else:
        right_curb_y = np.inf
    
    main_road = main_road[(main_road[:, 1] > left_curb_y) & (main_road[:, 1] < right_curb_y)]
    
    # Enhanced obstacle detection with classification
    proj = parse_calib(calib_path)
    img = cv2.imread(str(img_path))
    print("classifying weights")
    detected_objects, img = detect_and_classify_obstacles(
        np.asarray(pcd.points), main_road, img, proj,
        model, preprocess, labels,
        height_threshold=0.4, min_cluster_size=10
    )
    
    # Project and draw road points (blue)
    if len(main_road) > 0:
        road_uv = project(main_road, proj)
        for u, v in road_uv:
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (u, v), 1, (255, 100, 0), -1)
    
    # Draw curb points
    if len(left_rough) > 0:
        rough_uv = project(left_rough, proj)
        for u, v in rough_uv:
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (u, v), 2, (0, 255, 255), -1)
    
    if len(right_rough) > 0:
        rough_uvr = project(right_rough, proj)
        for u, v in rough_uvr:
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (u, v), 2, (0, 255, 0), -1)

    print(f"Processed {frame}: {len(main_road)} road points, {len(detected_objects)} detected objects")
    for obj in detected_objects:
        print(f"  - {obj['class']}: {obj['confidence']:.3f}")
    
    cv2.imshow('Enhanced Road Detection with Object Classification', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save output
    script_dir = Path(__file__).parent
    output_dir = script_dir / "enhanced_detection"
    output_dir.mkdir(exist_ok=True)
    output_img_path = output_dir / f"{frame}_enhanced_detection.png"
    cv2.imwrite(str(output_img_path), img)
    print(f"Saved enhanced detection result at {output_img_path}")
    
    return main_road, detected_objects

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