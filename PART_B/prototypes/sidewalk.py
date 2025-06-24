import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_erosion, binary_dilation
import cv2
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
def pca_high_variation_mask(points, radius=0.3, z_variance_thresh=0.002):
    """
    Return a boolean mask of points with high vertical roughness in local patch.
    Useful for excluding curb-like areas from road points.
    """
    kdt = cKDTree(points[:, :2])
    high_variance_mask = np.zeros(len(points), dtype=bool)
    
    for i, p in enumerate(points):
        idx = kdt.query_ball_point(p[:2], radius) 
        if len(idx) < 5:
            continue
        patch = points[idx]
        pca = PCA(n_components=3).fit(patch)
        # The 3rd eigenvalue (variance) is largest if there's a jump/step
        if pca.explained_variance_[2] > z_variance_thresh:
            high_variance_mask[i] = True
    return high_variance_mask

def project(pts, P):
    """Project 3D points to image coordinates"""
    h = np.hstack([pts, np.ones((len(pts), 1))])
    uvw = (P @ h.T).T
    z = uvw[:, 2]
    valid = z > 0
    uv = np.zeros((len(pts), 2), dtype=int)
    uv[valid] = (uvw[valid, :2] / z[valid, np.newaxis]).astype(int)
    return uv[valid]
def method1_skeleton_centerline(road_points, proj, img_shape):
    """
    Method 1: Morphological skeleton approach using standard OpenCV
    Creates a binary mask and extracts the skeleton as centerline
    """
    # Project points to image
    uv = project(road_points, proj)
    mask = np.zeros(img_shape[:2], np.uint8)
    
    # Create filled road mask
    for u, v in uv:
        if 0 <= u < img_shape[1] and 0 <= v < img_shape[0]:
            mask[v, u] = 255
    
    # Dilate to fill gaps, then get skeleton
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    

    skeleton = skeletonize(mask / 255).astype(np.uint8)  # Normalize to binary
    
    # Extract centerline points
    centerline_pixels = np.column_stack(np.where(skeleton > 0))
    return centerline_pixels[:, [1, 0]]  # Return as (u, v)

def zhang_suen_thinning(image):
    """
    Zhang-Suen thinning algorithm implementation using only OpenCV
    """
    # Convert to binary
    _, binary = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)
    
    changing1 = changing2 = 1
    while changing1 or changing2:
        # Step 1
        changing1 = []
        rows, columns = binary.shape
        for i in range(1, rows - 1):
            for j in range(1, columns - 1):
                if binary[i,j] == 1:
                    # Get 8-neighbors
                    p2 = binary[i-1, j]
                    p3 = binary[i-1, j+1]
                    p4 = binary[i, j+1]
                    p5 = binary[i+1, j+1]
                    p6 = binary[i+1, j]
                    p7 = binary[i+1, j-1]
                    p8 = binary[i, j-1]
                    p9 = binary[i-1, j-1]
                    
                    # Calculate conditions
                    A = (p2 == 0 and p3 == 1) + (p3 == 0 and p4 == 1) + \
                        (p4 == 0 and p5 == 1) + (p5 == 0 and p6 == 1) + \
                        (p6 == 0 and p7 == 1) + (p7 == 0 and p8 == 1) + \
                        (p8 == 0 and p9 == 1) + (p9 == 0 and p2 == 1)
                    B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                    
                    if (A == 1 and (B >= 2 and B <= 6) and 
                        (p2 * p4 * p6) == 0 and (p4 * p6 * p8) == 0):
                        changing1.append((i,j))
        
        for i, j in changing1:
            binary[i, j] = 0
            
        # Step 2
        changing2 = []
        for i in range(1, rows - 1):
            for j in range(1, columns - 1):
                if binary[i,j] == 1:
                    # Get 8-neighbors
                    p2 = binary[i-1, j]
                    p3 = binary[i-1, j+1]
                    p4 = binary[i, j+1]
                    p5 = binary[i+1, j+1]
                    p6 = binary[i+1, j]
                    p7 = binary[i+1, j-1]
                    p8 = binary[i, j-1]
                    p9 = binary[i-1, j-1]
                    
                    # Calculate conditions
                    A = (p2 == 0 and p3 == 1) + (p3 == 0 and p4 == 1) + \
                        (p4 == 0 and p5 == 1) + (p5 == 0 and p6 == 1) + \
                        (p6 == 0 and p7 == 1) + (p7 == 0 and p8 == 1) + \
                        (p8 == 0 and p9 == 1) + (p9 == 0 and p2 == 1)
                    B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                    
                    if (A == 1 and (B >= 2 and B <= 6) and 
                        (p2 * p4 * p8) == 0 and (p2 * p6 * p8) == 0):
                        changing2.append((i,j))
        
        for i, j in changing2:
            binary[i, j] = 0
    
    return (binary * 255).astype(np.uint8)

def method1_alternative_medial_axis(road_points, proj, img_shape):
    """
    Alternative Method 1: Using distance transform + local maxima
    Faster alternative to skeleton
    """
    from scipy.ndimage import distance_transform_edt
    from scipy.ndimage import maximum_filter
    
    # Project points to image
    uv = project(road_points, proj)
    mask = np.zeros(img_shape[:2], np.uint8)
    
    # Create filled road mask
    for u, v in uv:
        if 0 <= u < img_shape[1] and 0 <= v < img_shape[0]:
            mask[v, u] = 255
    
    # Fill gaps
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Distance transform
    binary_mask = mask > 0
    dist_transform = distance_transform_edt(binary_mask)
    
    # Find local maxima (centerline)
    local_maxima = maximum_filter(dist_transform, size=5) == dist_transform
    centerline = local_maxima & (dist_transform > 1)  # Remove thin areas
    
    # Extract centerline points
    centerline_pixels = np.column_stack(np.where(centerline))
    return centerline_pixels[:, [1, 0]]  # Return as (u, v)

def method2_pca_based_centerline(road_points):
    """
    Method 2: PCA-based approach in 3D space
    Fits a line through the road points and creates centerline
    """
    if len(road_points) < 10:
        return np.array([])
    
    # Remove outliers using statistical filtering
    mean_y = np.mean(road_points[:, 1])
    std_y = np.std(road_points[:, 1])
    filtered_points = road_points[np.abs(road_points[:, 1] - mean_y) < 2 * std_y]
    
    # PCA to find main road direction
    pca = PCA(n_components=2)
    pca.fit(filtered_points[:, [0, 1]])  # Use X, Y coordinates
    
    # Get road bounds
    x_min, x_max = filtered_points[:, 0].min(), filtered_points[:, 0].max()
    
    # Create centerline points along the main direction
    num_points = int((x_max - x_min) / 0.5)  # Every 0.5m
    x_line = np.linspace(x_min, x_max, num_points)
    
    # Project onto PCA line (assuming road center is at mean Y)
    centerline_3d = []
    for x in x_line:
        # Find points near this X coordinate
        nearby_mask = np.abs(filtered_points[:, 0] - x) < 1.0
        if np.any(nearby_mask):
            nearby_points = filtered_points[nearby_mask]
            y_center = np.median(nearby_points[:, 1])
            z_center = np.median(nearby_points[:, 2])
            centerline_3d.append([x, y_center, z_center])
    
    return np.array(centerline_3d)

def method3_sliding_window_centerline(road_points, window_size=2.0):
    """
    Method 3: Sliding window approach
    Divides road into segments and finds center of each segment
    """
    if len(road_points) < 10:
        return np.array([])
    
    # Sort by X coordinate (assuming road runs along X axis)
    sorted_points = road_points[np.argsort(road_points[:, 0])]
    
    x_min, x_max = sorted_points[:, 0].min(), sorted_points[:, 0].max()
    centerline_3d = []
    
    # Sliding window along X axis
    current_x = x_min
    while current_x < x_max:
        # Get points in current window
        window_mask = (sorted_points[:, 0] >= current_x) & (sorted_points[:, 0] < current_x + window_size)
        window_points = sorted_points[window_mask]
        
        if len(window_points) > 5:
            # Find Y boundaries (left and right edges)
            y_sorted = np.sort(window_points[:, 1])
            
            # Remove outliers (bottom and top 10%)
            trim_idx = int(len(y_sorted) * 0.1)
            if trim_idx > 0:
                y_trimmed = y_sorted[trim_idx:-trim_idx]
            else:
                y_trimmed = y_sorted
            
            # Center is median of trimmed Y values
            y_center = np.median(y_trimmed)
            z_center = np.median(window_points[:, 2])
            x_center = current_x + window_size / 2
            
            centerline_3d.append([x_center, y_center, z_center])
        
        current_x += window_size / 2  # 50% overlap
    
    return np.array(centerline_3d)

def method4_road_edges_centerline(road_points, eps=1.0, min_samples=10):
    """
    Method 4: Find road edges first, then compute centerline
    """
    if len(road_points) < 20:
        return np.array([])
    
    # Cluster points by Y coordinate to find left and right edges
    y_coords = road_points[:, 1].reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(y_coords)
    
    unique_labels = np.unique(clustering.labels_)
    valid_clusters = unique_labels[unique_labels != -1]
    
    if len(valid_clusters) < 2:
        return method3_sliding_window_centerline(road_points)
    
    # Get the two extreme clusters (leftmost and rightmost)
    cluster_centers = []
    for label in valid_clusters:
        cluster_points = road_points[clustering.labels_ == label]
        y_center = np.mean(cluster_points[:, 1])
        cluster_centers.append((label, y_center))
    
    cluster_centers.sort(key=lambda x: x[1])  # Sort by Y coordinate
    left_label, right_label = cluster_centers[0][0], cluster_centers[-1][0]
    
    left_edge = road_points[clustering.labels_ == left_label]
    right_edge = road_points[clustering.labels_ == right_label]
    
    # Create centerline between edges
    x_min = max(left_edge[:, 0].min(), right_edge[:, 0].min())
    x_max = min(left_edge[:, 0].max(), right_edge[:, 0].max())
    
    centerline_3d = []
    num_points = int((x_max - x_min) / 0.5)
    x_line = np.linspace(x_min, x_max, num_points)
    
    for x in x_line:
        # Find corresponding points on left and right edges
        left_nearby = left_edge[np.abs(left_edge[:, 0] - x) < 1.0]
        right_nearby = right_edge[np.abs(right_edge[:, 0] - x) < 1.0]
        
        if len(left_nearby) > 0 and len(right_nearby) > 0:
            left_y = np.median(left_nearby[:, 1])
            right_y = np.median(right_nearby[:, 1])
            center_y = (left_y + right_y) / 2
            
            left_z = np.median(left_nearby[:, 2])
            right_z = np.median(right_nearby[:, 2])
            center_z = (left_z + right_z) / 2
            
            centerline_3d.append([x, center_y, center_z])
    
    return np.array(centerline_3d)

def improved_remove_sidewalks(road_points, proj, img_shape, min_candidates=300, method='sliding_window'):
    """
    Improved version using better centerline detection
    """
    # 1. Cluster road segments
    road_clusters = cluster_road_segments(road_points, eps=1.5, min_samples=15)
    
    if not road_clusters:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3))
    
    main_road = max(road_clusters, key=len)
    main_road = main_road[np.abs(main_road[:, 1]) < 5.0]
    
    # 2. PCA curb detection
    pca_curb_mask = pca_high_variation_mask(main_road, radius=0.3, z_variance_thresh=0.0001)
    rough_points = main_road[pca_curb_mask]
    main_road_filtered = main_road[~pca_curb_mask]
    
    # 3. Get centerline using selected method
    if method == 'skeleton':
        centerline_pixels = method1_skeleton_centerline(main_road_filtered, proj, img_shape)
        centerline_3d = None
    elif method == 'pca':
        centerline_3d = method2_pca_based_centerline(main_road_filtered)
        centerline_pixels = None
    elif method == 'sliding_window':
        centerline_3d = method3_sliding_window_centerline(main_road_filtered)
        centerline_pixels = None
    elif method == 'edges':
        centerline_3d = method4_road_edges_centerline(main_road_filtered)
        centerline_pixels = None
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 4. Classify curb points
    if centerline_pixels is not None:
        # Image-space classification
        left_rough, right_rough = classify_by_image_centerline(
            rough_points, proj, img_shape, centerline_pixels
        )
    else:
        # 3D space classification
        left_rough, right_rough = classify_by_3d_centerline(
            rough_points, centerline_3d
        )
    
    # 5. Refine main road
    if len(left_rough) > min_candidates:
        left_curb_y = np.percentile(left_rough[:, 1], 95)
    else:
        left_curb_y = -np.inf
    
    if len(right_rough) > min_candidates:
        right_curb_y = np.percentile(right_rough[:, 1], 5)
    else:
        right_curb_y = np.inf
    
    main_road_refined = main_road_filtered[
        (main_road_filtered[:, 1] > left_curb_y) & 
        (main_road_filtered[:, 1] < right_curb_y)
    ]
    
    return main_road_refined, left_rough, right_rough

def classify_by_3d_centerline(points, centerline_3d):
    """
    Classify points as left/right based on 3D centerline
    """
    if len(centerline_3d) == 0 or len(points) == 0:
        return np.array([]), np.array([])
    
    left_points, right_points = [], []
    
    for point in points:
        # Find closest centerline point
        distances = np.sqrt(np.sum((centerline_3d[:, :2] - point[:2])**2, axis=1))
        closest_idx = np.argmin(distances)
        closest_center = centerline_3d[closest_idx]
        
        # Determine if point is left or right of centerline
        if point[1] < closest_center[1]:  # Y coordinate comparison
            left_points.append(point)
        else:
            right_points.append(point)
    
    return np.array(left_points), np.array(right_points)

def classify_by_image_centerline(points, proj, img_shape, centerline_pixels):
    """
    Classify points as left/right based on image centerline
    """
    uv = project(points, proj)
    left_points, right_points = [], []
    
    for point, (u, v) in zip(points, uv):
        if 0 <= v < img_shape[0] and 0 <= u < img_shape[1]:
            # Find closest centerline pixel
            if len(centerline_pixels) > 0:
                distances = np.sqrt(np.sum((centerline_pixels - [u, v])**2, axis=1))
                closest_center_u = centerline_pixels[np.argmin(distances), 0]
                
                if u < closest_center_u:
                    left_points.append(point)
                else:
                    right_points.append(point)
    
    return np.array(left_points), np.array(right_points)


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