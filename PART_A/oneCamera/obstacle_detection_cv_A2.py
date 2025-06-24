
#no YOLO


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
from road_detector_A1 import detect_road,split_lanes
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
    
def merge_bounding_boxes(boxes, overlap_thresh=0.2):
    """Simple NMS-style merge for axis-aligned boxes."""
    if not boxes:
        return []
    boxes = np.array(boxes)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = areas.argsort()[::-1]
    merged = []
    while order.size:
        i = order[0]
        merged.append([x1[i], y1[i], x2[i], y2[i]])
        # compute IoU with the rest
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        keep = np.where(iou <= overlap_thresh)[0]
        order = order[keep + 1]
    return merged

def parser(name):
    calib_path = 'C:\\Users\\USER\\Documents\\_CAMERA_LIDAR\\data_road\\training\\calib'
    left_img_path = 'C:\\Users\\USER\\Documents\\_CAMERA_LIDAR\\data_road\\training\\image_2'
    right_img_path = 'C:\\Users\\USER\\Documents\\_CAMERA_LIDAR\\data_road_right\\training\\image_3'
    filename=name+'.txt'

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



def obstacle_detection(frame_id="um_000040"):
    print("[INFO] Starting obstacle detection pipeline...")

    # 1) Calibration
    Q, left_img, right_img = parser(frame_id)

    # 2) Disparity → 3D points
    disp = compute_disparity_map(left_img, right_img)
    disp = cv2.medianBlur(disp, 5)
    mask3d = disp > 0
    pts3d = cv2.reprojectImageTo3D(disp, Q)[mask3d]
    pix3d = np.column_stack(np.where(mask3d))

    # limit points
    if pts3d.shape[0] > 80000:
        idx = np.random.choice(pts3d.shape[0], 80000, replace=False)
        pts3d, pix3d = pts3d[idx], pix3d[idx]

    # 3) RANSAC ground plane
    _, _, plane_model, _ = ransac_ground_3d(pts3d)

    # 4) Height filter
    obst_pts, obst_mask = filter_by_height(pts3d, plane_model, threshold=0.5)
    obst_pix = pix3d[obst_mask]

    # 5) Depth filter
    keep = obst_pts[:,2] < 30
    obst_pts, obst_pix = obst_pts[keep], obst_pix[keep]

    # 6) Road mask 
    out, road_mask, hull, gc_mask = detect_road(left_img, None)
    laneimage=split_lanes(left_img, road_mask, hull, gc_mask)
    road_bool = road_mask.astype(bool)
    #dilate road mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    dil = cv2.dilate(road_mask.astype(np.uint8), kernel, iterations=2)
    road_bool = dil.astype(bool)
    # 7) DBSCAN clustering on all obstacle points
    if obst_pts.shape[0] > 10000:
        idx = np.random.choice(obst_pts.shape[0], 10000, replace=False)
        obst_pts, obst_pix = obst_pts[idx], obst_pix[idx]

    labels = DBSCAN(eps=0.6, min_samples=30) \
                .fit_predict(obst_pts[:, [0,2]])

    # 8) Post-filter clusters by overlap with original mask
    raw_boxes = []
    for cid in np.unique(labels):
        if cid == -1:
            continue
        pix = obst_pix[labels == cid]
        if pix.shape[0] < 40:
            continue
        # overlap on original mask
        on_road = np.count_nonzero(road_bool[pix[:,0], pix[:,1]])
        if on_road / pix.shape[0] < 0.05:
            continue
        y0, x0 = pix.min(axis=0)
        y1, x1 = pix.max(axis=0)
        raw_boxes.append([x0, y0, x1, y1])

    # 9) Merge overlapping boxes
    merged_boxes = merge_bounding_boxes(raw_boxes, overlap_thresh=0.2)

    # 10) Draw merged boxes
    for x0, y0, x1, y1 in merged_boxes:
        cv2.rectangle(left_img, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.rectangle(laneimage, (x0, y0), (x1, y1), (0, 0, 255), 2)
    # 11) Show original road region as a translucent overlay
    overlay = left_img.copy()
    colored = np.zeros_like(left_img)          # Red
    colored[road_bool] = (255, 0, 0)            # Blue tint for road
    cv2.addWeighted(colored, 0.3, overlay, 0.7, 0, overlay)
    cv2.imshow("Road Mask Overlay", overlay)


    coloredinlane= np.zeros_like(left_img)
    colored[road_bool] = (0, 0, 255)  
    cv2.addWeighted(coloredinlane, 0.3, laneimage, 0.7, 0, laneimage)   
    cv2.imshow("Lane Image", laneimage)

    # 12) Final displays
    # cv2.imshow("Obstacle Detections", left_img)
    # disp_vis = (disp - disp.min()) / (disp.max() - disp.min())
    # cv2.imshow("Disparity", disp_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_on_all_frames():
    calib_path = 'C:\\Users\\USER\\Documents\\_CAMERA_LIDAR\\data_road\\training\\calib'
    for filename in os.listdir(calib_path):
        if filename.endswith(".txt"):
            frame_id = os.path.splitext(filename)[0]
            print(f"[INFO] Processing frame: {frame_id}")
            try:
                obstacle_detection(frame_id)
            except Exception as e:
                print(f"[ERROR] Failed to process {frame_id}: {e}")
def run_selected_frames():
    calib_path = 'C:\\Users\\USER\\Documents\\_CAMERA_LIDAR\\data_road\\training\\calib'
    selected_frames = ["um_000039" ,"umm_000008","umm_000007","um_000040","um_000072", "uu_000081"]
    for frame_id in selected_frames:
        print(f"[INFO] Processing frame: {frame_id}")
        try:
            obstacle_detection(frame_id)
        except Exception as e:
            print(f"[ERROR] Failed to process {frame_id}: {e}")

if __name__ == "__main__":
    from road_detector_A1 import overlay_hull
    run_on_all_frames()
    run_selected_frames()
    #main_badram()
    # names=["um_000040","um_000072"]
    # name="uu_000081"
    # for name in names:
    #     obstacle_detection(frame_id=name)


