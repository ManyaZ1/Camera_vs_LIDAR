#add YOLO


#✅ Pipeline Ανίχνευσης Εμποδίων στον Δρόμο 
#1. Disparity Map από stereo εικόνες
'Χρησιμοποίησε cv2.StereoSGBM ή cv2.StereoBM.'

'Μετατροπή του disparity σε 3D point cloud (XYZ) με reprojectImageTo3D() ή custom Q matrix.'

#2. RANSAC για εύρεση επιπέδου του εδάφους
'Εφάρμοσε RANSAC σε 3D σημεία για να βρεις το ground plane.'

'Μοντέλο: ax + by + cz + d = 0.'

#3. Αφαίρεση του επιπέδου εδάφους
'Υπολόγισε την κάθετη απόσταση κάθε σημείου από το επίπεδο.'

'Κράτησε μόνο τα σημεία πάνω από το έδαφος (π.χ. > 0.2m).'

#4. DBSCAN clustering σε 3D
'Ομαδοποίησε τα σημεία που είναι πάνω από το έδαφος.'

'Κάθε cluster θεωρείται "εμπόδιο".'

#5. Προβολή bounding boxes στην εικόνα
'Για κάθε cluster, υπολόγισε το 2D bounding box από τα σημεία του.'

'Προέβαλέ το στην εικόνα αν θέλεις visual feedback.'

## PIPELINE 2 ##
#1.pre process
#2.stereo camera calibration
#3.Road Surface Detection with ransac
#4.cluster with dbscan
#5.bounding box projection ?
#6.visualization
import numpy as np
import cv2
import os
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from road_detector_A1 import detect_road
def parse_kitti_calib(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    
    # P2 = για την αριστερή κάμερα (KITTI: rectified left)
    # P3 = για τη δεξιά κάμερα
    P2 = np.array([float(val) for val in lines[2].split()[1:]]).reshape(3, 4)
    P3 = np.array([float(val) for val in lines[3].split()[1:]]).reshape(3, 4)

    fx = P2[0, 0]
    cx = P2[0, 2]
    cy = P2[1, 2]
    Tx2 = P2[0, 3]
    Tx3 = P3[0, 3]
    Tx = -(Tx3 - Tx2) / fx  # baseline in meters

    # Q matrix για reprojectImageTo3D
    Q = np.array([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0, fx],
        [0, 0, -1 / Tx, 0]
    ], dtype=np.float32)

    return Q
def parse_kitti_calib1(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    
    P2 = np.array([float(val) for val in lines[2].split()[1:]]).reshape(3, 4)
    P3 = np.array([float(val) for val in lines[3].split()[1:]]).reshape(3, 4)
    
    fx = P2[0, 0]
    cx = P2[0, 2]
    cy = P2[1, 2]
    Tx = -(P3[0, 3] - P2[0, 3]) / fx

    Q = np.array([
        [1, 0,   0,   -cx],
        [0, 1,   0,   -cy],
        [0, 0,   0,    fx],
        [0, 0, -1/Tx,   0]
    ])
    
    return Q


def filter_by_height(points, plane_model, threshold=0.3):
    a, b, c, d = plane_model
    numerator = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)
    denominator = np.sqrt(a**2 + b**2 + c**2)
    dist = numerator / denominator
    return points[dist > threshold], dist > threshold


def compute_disparity_map(left_img, right_img):
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,   # multiple of 16
        blockSize=5,
        P1=8 * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2
    )

    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    return disparity

import open3d as o3d
import numpy as np

def ransac_ground_3d(points: np.ndarray,
                     distance_threshold: float = 0.01,
                     ransac_n: int = 3,
                     num_iterations: int = 2000,
                     show: bool = False) -> tuple:
    """
    Εκτελεί RANSAC με Open3D για να αφαιρέσει το έδαφος από 3D σημεία.
    Επιστρέφει και τα indices των inliers για να τα συσχετίσεις με pixels.

    Returns:
    - obstacle_points: points not on the plane
    - ground_points: points on the plane
    - plane_model: (a, b, c, d)
    - ground_indices: indices of ground points (inliers)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )

    ground_points = pcd.select_by_index(inliers)
    obstacle_points = pcd.select_by_index(inliers, invert=True)

    if show:
        ground_points.paint_uniform_color([0, 1, 0])
        obstacle_points.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([ground_points, obstacle_points])

    return (
        np.asarray(obstacle_points.points),
        np.asarray(ground_points.points),
        plane_model,
        inliers  # <== αυτό προστέθηκε
    )

def ransac_plane_fit(points, threshold=0.01):
    """
    Εφαρμόζει RANSAC για να εντοπίσει επίπεδο εδάφους σε 3D σημεία.

    Parameters:
    - points: Nx3 πίνακας με [X, Y, Z]
    - threshold: αποδεκτή απόσταση (π.χ. 0.01m) για inliers

    Returns:
    - plane_model: (a, b, c, d) coefficients of the plane
    - inlier_mask: boolean mask των σημείων που ανήκουν στο επίπεδο
    """
    X = points[:, [0, 2]]  # Χ και Ζ
    y = points[:, 1]       # Υ (ύψος)
    model = RANSACRegressor(
    residual_threshold=0.01,  # ή 0.02 (ανάλογα το scale των μονάδων σου)
    max_trials=1000,
    min_samples=0.5  # πιο αυστηρό
)

    #model = RANSACRegressor(residual_threshold=threshold,max_trials=1000, min_samples=0.2)
    model.fit(X, y)
    inlier_mask = model.inlier_mask_

    a, c = model.estimator_.coef_
    b = -1.0
    d = model.estimator_.intercept_

    # ax + by + cz + d = 0 => λύνεται για y = ax + cz + d
    return (a, b, c, d), inlier_mask
def filter_by_height(points, plane_model, threshold=0.2):
    a, b, c, d = plane_model
    numerator = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)
    denominator = np.sqrt(a ** 2 + b ** 2 + c ** 2)
    dist = numerator / denominator
    mask = dist > threshold
    return points[mask], mask


def main_badram():
    print("[INFO] Starting obstacle detection pipeline...")
    # === Βήμα 1: Paths ===
    calib_path = 'C:\\Users\\USER\\Documents\\_CAMERA_LIDAR\\data_road\\training\\calib'
    left_img_path = 'C:\\Users\\USER\\Documents\\_CAMERA_LIDAR\\data_road\\training\\image_2\\um_000040.png'
    right_img_path = 'C:\\Users\\USER\\Documents\\_CAMERA_LIDAR\\data_road_right\\training\\image_3\\um_000040.png'

    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)

    if left_img is None or right_img is None:
        print("[ERROR] Could not load stereo images.")
        return

    # === Βήμα 2: Calibration ===
    Q = parse_kitti_calib(os.path.join(calib_path, 'um_000040.txt'))

    # === Βήμα 3: Disparity Map ===
    disparity = compute_disparity_map(left_img, right_img)
    disparity = cv2.medianBlur(disparity, 5)

    # === Βήμα 4: Reproject to 3D ===
    mask = disparity > 0
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    valid_points = points_3D[mask]  # (N, 3)
    pixel_coords = np.column_stack(np.where(mask))  # (N, 2)

    # === Βήμα 5: RANSAC για ground plane ===
    obstacle_pts, ground_pts, plane_model, ground_indices = ransac_ground_3d(valid_points)

    # === Βήμα 6: Φιλτράρισμα based on height above plane ===
    obstacle_points, obstacle_mask = filter_by_height(valid_points, plane_model, threshold=0.5)
    obstacle_pixels = pixel_coords[obstacle_mask]
    #Αγνόησε clusters πολύ μακριά (βάθος Z):
    max_depth = 25  # π.χ. αγνόησε ό,τι είναι πέρα από 25 μέτρα
    keep_mask = obstacle_points[:, 2] < max_depth
    obstacle_points = obstacle_points[keep_mask]
    obstacle_pixels = obstacle_pixels[keep_mask]
    #add road mask
    print("[INFO] detecting road")
    out, road_hull_mask, hull, gc_mask = detect_road(left_img)
    print("[INFO] Road mask computed.")
    road_mask_values = road_hull_mask[obstacle_pixels[:, 0], obstacle_pixels[:, 1]]  # boolean mask
    obstacle_points = obstacle_points[road_mask_values]
    obstacle_pixels = obstacle_pixels[road_mask_values]
    # === Βήμα 7: DBSCAN clustering ===
    #labels = DBSCAN(eps=0.4, min_samples=30).fit_predict(obstacle_points)
    labels = DBSCAN(eps=0.4, min_samples=50).fit_predict(obstacle_points[:, [0, 2]])
    # === Βήμα 8: Προβολή bounding boxes ===
    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            continue  # Skip noise

        cluster_mask = labels == cluster_id
        cluster_pixels = obstacle_pixels[cluster_mask]

        if len(cluster_pixels) < 40:
            continue

        y_min, x_min = np.min(cluster_pixels, axis=0)
        y_max, x_max = np.max(cluster_pixels, axis=0)

        cv2.rectangle(left_img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    # === Εμφάνιση αποτελεσμάτων ===
    cv2.imshow("Obstacle Detections", left_img)
    cv2.imshow("Disparity", (disparity - disparity.min()) / (disparity.max() - disparity.min()))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def tester():
    calib_path = 'C:\\Users\\USER\\Documents\\_CAMERA_LIDAR\\data_road\\training\\calib'
    left_img_path = 'C:\\Users\\USER\\Documents\\_CAMERA_LIDAR\\data_road\\training\\image_2'
    right_img_path = 'C:\\Users\\USER\\Documents\\_CAMERA_LIDAR\\data_road_right\\training\\image_3'
    for filename in os.listdir(calib_path):
        if filename.lower().endswith(('.txt')):
            imgname= filename.split('.')[0]
            left_img_path = os.path.join(left_img_path, imgname + '.png')
            right_img_path = os.path.join(right_img_path, imgname + '.png')
            
            left_img = cv2.imread(left_img_path)
            right_img = cv2.imread(right_img_path)

            if left_img is None or right_img is None:
                print("[ERROR] Could not load stereo images.")

            # === Βήμα 2: Calibration ===
            Q = parse_kitti_calib(os.path.join(calib_path, filename))
            return Q, left_img, right_img
def main():
    print("[INFO] Starting obstacle detection pipeline...")
    # === Βήμα 1: Paths ===
    # calib_path = 'C:\\Users\\USER\\Documents\\_CAMERA_LIDAR\\data_road\\training\\calib'
    # left_img_path = 'C:\\Users\\USER\\Documents\\_CAMERA_LIDAR\\data_road\\training\\image_2\\um_000042.png'
    # right_img_path = 'C:\\Users\\USER\\Documents\\_CAMERA_LIDAR\\data_road_right\\training\\image_3\\um_000042.png'

    # left_img = cv2.imread(left_img_path)
    # right_img = cv2.imread(right_img_path)

    # if left_img is None or right_img is None:
    #     print("[ERROR] Could not load stereo images.")
    #     return

    # # === Βήμα 2: Calibration ===
    # Q = parse_kitti_calib(os.path.join(calib_path, 'um_000042.txt'))
    Q,left_img,right_img=tester()
    # === Βήμα 3: Disparity Map ===
    disparity = compute_disparity_map(left_img, right_img)
    disparity = cv2.medianBlur(disparity, 5)

    # === Βήμα 4: Reproject to 3D ===
    mask = disparity > 0
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    valid_points = points_3D[mask]
    pixel_coords = np.column_stack(np.where(mask))

    # === Περιορισμός πλήθους για αποφυγή RAM crash ===
    if len(valid_points) > 80000:
        indices = np.random.choice(len(valid_points), 80000, replace=False)
        valid_points = valid_points[indices]
        pixel_coords = pixel_coords[indices]

    # === Βήμα 5: RANSAC για ground plane ===
    obstacle_pts, ground_pts, plane_model, ground_indices = ransac_ground_3d(valid_points)

    # === Βήμα 6: Φιλτράρισμα based on height above plane ===
    obstacle_points, obstacle_mask = filter_by_height(valid_points, plane_model, threshold=0.5)
    obstacle_pixels = pixel_coords[obstacle_mask]

    # === Φιλτράρισμα με βάση βάθος (Z) ===
    max_depth = 30
    keep_mask = obstacle_points[:, 2] < max_depth
    obstacle_points = obstacle_points[keep_mask]
    obstacle_pixels = obstacle_pixels[keep_mask]

    # === Βήμα 7: Εφαρμογή road mask ===
    print("[INFO] Detecting road...")
    out, road_hull_mask, hull, gc_mask = detect_road(left_img,None)
    h, w = road_hull_mask.shape

    # Ασφαλής επιλογή pixel που βρίσκονται εντός εικόνας
    in_bounds = (obstacle_pixels[:, 0] < h) & (obstacle_pixels[:, 1] < w)
    obstacle_pixels = obstacle_pixels[in_bounds]
    obstacle_points = obstacle_points[in_bounds]

    # # Εφαρμογή μάσκας δρόμου
    road_mask_values = road_hull_mask[obstacle_pixels[:, 0], obstacle_pixels[:, 1]]
    kernel = np.ones((20, 20), np.uint8)  # You can tune this
    road_hull_mask_dilated = cv2.dilate(road_hull_mask.astype(np.uint8), kernel, iterations=3)
    road_hull_mask_dilated = road_hull_mask_dilated.astype(bool)
    road_mask_values = road_hull_mask_dilated[obstacle_pixels[:, 0], obstacle_pixels[:, 1]]
    obstacle_pixels = obstacle_pixels[road_mask_values]
    obstacle_points = obstacle_points[road_mask_values]

    # === Βήμα 8: DBSCAN clustering ===
    if len(obstacle_points) > 10000:
        idx = np.random.choice(len(obstacle_points), 10000, replace=False)
        obstacle_points = obstacle_points[idx]
        obstacle_pixels = obstacle_pixels[idx]

    labels = DBSCAN(eps=0.5, min_samples=50).fit_predict(obstacle_points[:, [0, 2]])

    # === Βήμα 9: Προβολή bounding boxes ===
    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            continue
        cluster_mask = labels == cluster_id
        cluster_pixels = obstacle_pixels[cluster_mask]

        if len(cluster_pixels) < 40:
            continue

        y_min, x_min = np.min(cluster_pixels, axis=0)
        y_max, x_max = np.max(cluster_pixels, axis=0)

        cv2.rectangle(left_img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    # === Εμφάνιση αποτελεσμάτων ===
    cv2.imshow("Obstacle Detections", left_img)
    cv2.imshow("Disparity", (disparity - disparity.min()) / (disparity.max() - disparity.min()))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()