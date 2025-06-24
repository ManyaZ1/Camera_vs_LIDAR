import cv2
import numpy as np
from obstacle_detection_cv_A2 import parser
# from obstacle_detection_cv_A2 import (
#     parser, compute_disparity_map, filter_by_height, ransac_ground_3d,
#     detect_road, merge_bounding_boxes
# )
#from road_detector_A1 import overlay_hull
import cv2
import numpy as np


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
def ransac_plane_numpy(points, num_iterations=100, distance_threshold=0.02):
    """
    Fits a plane to 3D points using RANSAC.
    
    Parameters:
        points (Nx3): 3D point cloud.
        num_iterations (int): How many RANSAC trials to run.
        distance_threshold (float): Max distance from plane to be considered an inlier.

    Returns:
        best_plane (a, b, c, d): Coefficients of the best plane.
        inlier_mask (bool array): Mask of points that lie close to the plane.
    """
    best_inliers = 0
    best_plane = None
    best_mask = None

    for _ in range(num_iterations):
        # 1. Randomly sample 3 unique points
        idx = np.random.choice(len(points), 3, replace=False)
        sample = points[idx]
        
        p1, p2, p3 = sample

        # 2. Compute the plane normal (cross product of two vectors)
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)

        if np.linalg.norm(normal) == 0:
            continue  # Degenerate case: points are collinear

        a, b, c = normal
        d = -np.dot(normal, p1)

        # 3. Compute distance of all points to the plane
        numer = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)
        denom = np.sqrt(a**2 + b**2 + c**2)
        dist = numer / denom

        # 4. Count inliers
        inlier_mask = dist < distance_threshold
        inliers = np.sum(inlier_mask)

        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = (a, b, c, d)
            best_mask = inlier_mask

    return best_plane, best_mask


# --- Load YOLO ---
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# --- Load Image ---
Q, left_img, right_img = parser("um_000019")
image = left_img #cv2.imread("um_000019.png")
h, w = image.shape[:2]

# --- Run YOLO ---
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward(output_layers)

# --- Process Detections ---
boxes, confidences, class_ids = [], [], []
conf_threshold = 0.5
nms_threshold = 0.4

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_threshold:
            cx, cy, bw, bh = detection[:4]
            x = int(cx * w - bw * w / 2)
            y = int(cy * h - bh * h / 2)
            bw = int(bw * w)
            bh = int(bh * h)
            boxes.append([x, y, bw, bh])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
yolo_boxes = []
for i in indices.flatten():
    x, y, bw, bh = boxes[i]
    x1, y1, x2, y2 = x, y, x + bw, y + bh
    yolo_boxes.append([x1, y1, x2, y2])

# --- Draw and Mask ---
image_with_boxes = image.copy()
image_removed = image.copy()

for i in indices.flatten():
    x, y, bw, bh = boxes[i]
    x1, y1, x2, y2 = x, y, x + bw, y + bh

    # Draw box on detection image
    cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
    cv2.putText(image_with_boxes, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Blackout on removed image
    image_removed[y1:y2, x1:x2] = 0

# --- Show in two windows ---
#cv2.imshow("YOLO Detected Obstacles", image_with_boxes)
#cv2.imshow("Obstacles Removed", image_removed)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 0. disparity map
Q, left_img, right_img = parser("um_000019")
disparity = compute_disparity_map(left_img,right_img)   

# 1. Generate 3D point cloud
points_3D = cv2.reprojectImageTo3D(disparity, Q)
mask_valid = disparity > 0

# 2. Filter out points inside YOLO boxes
mask_yolo = np.ones(disparity.shape, dtype=np.uint8)
for x1, y1, x2, y2 in yolo_boxes:
    mask_yolo[y1:y2, x1:x2] = 0
mask_filtered = (mask_valid & (mask_yolo > 0))

# 3. Extract filtered 3D points
filtered_points = points_3D[mask_filtered]

# 4. Run RANSAC
np.random.seed(0)  
plane_model, inlier_mask = ransac_plane_numpy(
        filtered_points,
        num_iterations=1000,      # 2) more iterations
        distance_threshold=0.03   # 3) slightly looser tol
)
# print("Plane coefficients:", plane_model)
a, b, c, d = plane_model
norm = np.linalg.norm([a, b, c])
a, b, c, d = a / norm, b / norm, c / norm, d / norm
if b < 0:                         # always keep normal pointing “up”
    a, b, c, d = -a, -b, -c, -d
plane_model = (a, b, c, d)
print("Normalised plane:", plane_model,
      " | #inliers:", inlier_mask.sum())

inlier_points = filtered_points[inlier_mask]
inlier_pixels = np.column_stack(np.where(mask_filtered))[inlier_mask]

vis = left_img.copy()
for y, x in inlier_pixels:
    cv2.circle(vis, (x, y), 1, (255, 0, 0), -1)

cv2.imshow("RANSAC Inliers on Image", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()

road_mask = np.zeros(disparity.shape, dtype=np.uint8)
inlier_pixels = np.column_stack(np.where(mask_filtered))[inlier_mask]
for y, x in inlier_pixels:
    road_mask[y, x] = 255

cv2.imshow("Road Mask", road_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

overlay = left_img.copy()
mask_layer = np.zeros_like(left_img)
mask_layer[road_mask == 255] = (255, 0, 0)  # solid blue where road is

blended = cv2.addWeighted(mask_layer, 0.4, left_img, 0.6, 0)
cv2.imshow("Road Overlay", blended)
cv2.waitKey(0)
cv2.destroyAllWindows()


## -- adding road detection
def build_plane_mask(plane_model, points_3D, mask_valid,
                     tol=0.08):             # metres: tune 5-10 cm
    a, b, c, d = plane_model
    den = np.linalg.norm([a, b, c])

    dist = np.abs(a*points_3D[...,0] +
                  b*points_3D[...,1] +
                  c*points_3D[...,2] + d) / den

    mask = (mask_valid & (dist < tol)).astype(np.uint8) * 255  # 0 / 255
    # Fill tiny gaps
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, 2)
    mask = cv2.dilate(mask, k, 2)
    return mask            # same H×W as disparity
import os
from road_detector_A1 import grab_cut,retinex_enhanced,build_trapezoid_roi, compute_c1_channel, region_growing, post_processingv1, overlay_hull, largest_hull   

def detect_road_new(img,plane_mask=None,imgpath=None,debug=False): 
    h, w = img.shape[:2]
    roi = build_trapezoid_roi((h, w), top_ratio=0.55)
    seeds = [(w//4, int(h*0.95)), (w//2, int(h*0.95)), (3*w//4, int(h*0.95))]

    # --- compute c1+CLAHE region growing ---
    c1 = compute_c1_channel(img)
    c1_roi = cv2.bitwise_and(c1, roi)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    c1 = clahe.apply(c1_roi)
    c1_rg = region_growing(c1, seeds, threshold=15)
    c1_rg = post_processingv1(c1_rg)
    mb,mg,mr = cv2.mean(img, mask=c1_rg)[:3]  # discard alpha
    if debug:
        print("c1: Average BGR in region-growing:", mb,mg,mr)
    
    # --- compute Lab + retinex region growing ---
    img_proc = retinex_enhanced(img, sigma=40)
    lab      = cv2.cvtColor(img_proc, cv2.COLOR_BGR2Lab)
    lab_ch = cv2.equalizeHist(lab[:, :, 2])
    lab_rg = region_growing(lab_ch, seeds, threshold=15)
    lab_rg = post_processingv1(lab_rg)
    if cv2.countNonZero(lab_rg) < 400:
        rg = cv2.dilate(lab_rg, np.ones((5,5), np.uint8), 1)
    mbl,mgl,mrl = cv2.mean(img, mask=lab_rg)[:3]  # discard alpha
    if debug:
        print("lab: Average BGR in region-growing:", mbl, mgl, mrl)
    tol=6
    if mbl>mgl and mbl>mrl:
        mode = 'lab'

    elif mgl>mbl+tol:
        mode = 'c1'
    elif mrl>mbl+tol :
        mode='c1'
    elif mgl>mrl+tol:
        mode='c1'
    else:
        mode = 'lab'    

    # --- apply selected or merged mask ---
    if mode == 'c1':
        print("[INFO] Using C1 mode")
        sel_im=img.copy()
        rg_mask = c1_rg
    elif mode == 'lab':
        print("[INFO] Using Lab mode")
        sel_im=img_proc
        rg_mask = lab_rg
    else:
        rg_mask = cv2.bitwise_or(c1_rg, lab_rg)

    if cv2.countNonZero(rg_mask) < 300:
        rg_mask = cv2.dilate(rg_mask, np.ones((5, 5), np.uint8), 2)

    # --- GrabCut refinement ---
    if plane_mask is not None:
        rg_mask = cv2.bitwise_and(rg_mask, plane_mask)    
    
    gc_mask = grab_cut(sel_im, rg_mask.copy(), iterations=15, strong_foreground_width_ratio=0.25)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gc_mask = cv2.morphologyEx(gc_mask, cv2.MORPH_CLOSE, kernel)
    gc_mask = cv2.dilate(gc_mask, kernel, iterations=3)
    gc_mask = cv2.bitwise_and(gc_mask, roi) 

    gc_mask = cv2.bitwise_and(gc_mask, plane_mask)  # keep only pixels close to plane

    #conect contours
    cnts, _ = cv2.findContours(gc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        print("[WARN] nothing detected"); return

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    if len(cnts) >= 2 and cv2.contourArea(cnts[1]) > 0.35 * cv2.contourArea(cnts[0]):
        all_pts = np.vstack([cnts[0], cnts[1]])
    else:
        all_pts = cnts[0]
        print("not merged")
    
    hull = cv2.convexHull(all_pts)
    
    mask_hull = np.zeros_like(gc_mask)
    cv2.fillPoly(mask_hull, [hull], 255)
    # --- convex hull extraction ---
    #big_cnt, hull = largest_hull(gc_mask) #????????
    print(f"[DEBUG] Hull area: {cv2.contourArea(hull) if hull is not None else 'None'}")
    if hull is None : #or cv2.contourArea(hull) < 400
        print('[WARN] Detected hull too small — trying fallback channel...')
        fallback_mode = 'lab' if mode == 'c1' else 'c1'
        fallback_rg = lab_rg if fallback_mode == 'lab' else c1_rg
        gc_mask = grab_cut(img.copy(), fallback_rg.copy(), iterations=10, strong_foreground_width_ratio=0.2)
        gc_mask = cv2.morphologyEx(gc_mask, cv2.MORPH_CLOSE, kernel)
        gc_mask = cv2.dilate(gc_mask, kernel, iterations=3)
        gc_mask = cv2.bitwise_and(gc_mask, roi)

        big_cnt, hull = largest_hull(gc_mask)
        if hull is None:
            print('[ERROR] Fallback also failed.')
            return
        mode = fallback_mode

    road_hull_mask = np.zeros_like(gc_mask)
    cv2.fillPoly(road_hull_mask, [hull], 255)

    # --- visualization ---
    out = img.copy()
    overlay_hull(out, hull, color=(255, 0, 0), alpha=0.35)

    # cv2.imshow('Region Growing ({} mode)'.format(mode), rg_mask)
    # cv2.imshow('GrabCut mask', gc_mask)
    if debug:
        cv2.imshow('Road hull overlay', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if imgpath:
        outpath = 'road_results'
        img_name = os.path.basename(imgpath).split('.')[0]
        os.makedirs(outpath, exist_ok=True)  # <-- απαραίτητο

        path = os.path.join(outpath, img_name + '_lanes.png')
        success = cv2.imwrite(path, out)

        if success:
            print(f"[INFO] Saved result to {path}")
        else:
            print(f"[ERROR] Failed to save image to {path}")
    return out, road_hull_mask, hull, gc_mask



plane_mask = build_plane_mask(plane_model, points_3D, mask_valid)
cv2.imshow("Plane mask", plane_mask)
out, road_hull_mask, hull, gc_mask = detect_road_new(left_img,
                                                 plane_mask=plane_mask,
                                                 debug=True)

























# # Load YOLO once
# net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# with open("coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]
# output_layers = [net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# def detect_yolo_boxes(image, conf_threshold=0.5, nms_threshold=0.4):
#     h, w = image.shape[:2]
#     blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
#     net.setInput(blob)
#     outputs = net.forward(output_layers)

#     boxes = []
#     confidences = []
#     for output in outputs:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > conf_threshold:
#                 cx, cy, bw, bh = detection[:4]
#                 x = int(cx * w - bw * w / 2)
#                 y = int(cy * h - bh * h / 2)
#                 bw = int(bw * w)
#                 bh = int(bh * h)
#                 boxes.append([x, y, bw, bh])
#                 confidences.append(float(confidence))

#     indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
#     final_boxes = []
#     for i in indices.flatten():
#         x, y, bw, bh = boxes[i]
#         final_boxes.append([x, y, x + bw, y + bh])  # [x1, y1, x2, y2]

#     return final_boxes


# def mask_out_boxes(h, w, boxes):
#     mask = np.ones((h, w), dtype=np.uint8)
#     for x1, y1, x2, y2 in boxes:
#         x1 = max(x1, 0)
#         y1 = max(y1, 0)
#         x2 = min(x2, w - 1)
#         y2 = min(y2, h - 1)
#         mask[y1:y2, x1:x2] = 0
#     return mask.astype(bool)
# import numpy as np

# def ransac_plane_numpy(points, num_iterations=100, distance_threshold=0.02):
#     """
#     Fits a plane to 3D points using RANSAC.
    
#     Parameters:
#         points (Nx3): 3D point cloud.
#         num_iterations (int): How many RANSAC trials to run.
#         distance_threshold (float): Max distance from plane to be considered an inlier.

#     Returns:
#         best_plane (a, b, c, d): Coefficients of the best plane.
#         inlier_mask (bool array): Mask of points that lie close to the plane.
#     """
#     best_inliers = 0
#     best_plane = None
#     best_mask = None

#     for _ in range(num_iterations):
#         # 1. Randomly sample 3 unique points
#         idx = np.random.choice(len(points), 3, replace=False)
#         sample = points[idx]
        
#         p1, p2, p3 = sample

#         # 2. Compute the plane normal (cross product of two vectors)
#         v1 = p2 - p1
#         v2 = p3 - p1
#         normal = np.cross(v1, v2)

#         if np.linalg.norm(normal) == 0:
#             continue  # Degenerate case: points are collinear

#         a, b, c = normal
#         d = -np.dot(normal, p1)

#         # 3. Compute distance of all points to the plane
#         numer = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)
#         denom = np.sqrt(a**2 + b**2 + c**2)
#         dist = numer / denom

#         # 4. Count inliers
#         inlier_mask = dist < distance_threshold
#         inliers = np.sum(inlier_mask)

#         if inliers > best_inliers:
#             best_inliers = inliers
#             best_plane = (a, b, c, d)
#             best_mask = inlier_mask

#     return best_plane, best_mask

# def main_yolo_filtered_ransac(frame_id="uu_000081"):
#     Q, left_img, right_img = parser(frame_id)
#     disp = compute_disparity_map(left_img, right_img)
#     disp = cv2.medianBlur(disp, 5)

#     mask_valid = disp > 0
#     h, w = disp.shape
#     pts3d = cv2.reprojectImageTo3D(disp, Q)
#     pix_coords = np.column_stack(np.where(mask_valid))
#     valid_pts = pts3d[mask_valid]

#     # Detect obstacles using YOLO
#     yolo_boxes = detect_yolo_boxes(left_img)

#     # Create mask for keeping only non-obstacle pixels
#     obstacle_mask = mask_out_boxes(h, w, yolo_boxes)
#     combined_mask = obstacle_mask[mask_valid]
#     filtered_pts = valid_pts[combined_mask]
#     filtered_pix = pix_coords[combined_mask]

#     print(f"[INFO] Total 3D points: {len(valid_pts)}, after YOLO removal: {len(filtered_pts)}")

#     # RANSAC to detect road plane
#     if len(filtered_pts) < 1000:
#         print("[WARNING] Not enough points after masking to fit plane.")
#         return
#     _, ground_pts, plane_model, _ = ransac_ground_3d(filtered_pts, show=True)

#     # Visualize YOLO boxes
#     vis = left_img.copy()
#     for x1, y1, x2, y2 in yolo_boxes:
#         cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv2.imshow("YOLO Detections", vis)
#     disp_vis = (disp - disp.min()) / (disp.max() - disp.min())
#     #cv2.imshow("Disparity", disp_vis)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main_yolo_filtered_ransac("uu_000081")
