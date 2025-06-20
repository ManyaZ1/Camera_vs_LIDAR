import argparse
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d

# ────────────────────────── CLI ────────────────────────── #
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Road plane extraction with obstacle masking (YOLO→RANSAC).")
    p.add_argument("--index", default="all", help="KITTI frame id without extension, e.g. um_000019")
    p.add_argument("--calib_dir", default="C:/Users/USER/Documents/_CAMERA_LIDAR/data_road/training/calib")
    p.add_argument("--left_dir",  default="C:/Users/USER/Documents/_CAMERA_LIDAR/data_road/training/image_2")
    p.add_argument("--right_dir", default="C:/Users/USER/Documents/_CAMERA_LIDAR/data_road_right/training/image_3")
    p.add_argument("--yolo_cfg", default="C:/Users/USER/Documents/GitHub/Camera_vs_LIDAR/PART_A/version2_dbsacan_yolo/yolov3.cfg")
    
    p.add_argument("--yolo_weights", default="C:/Users/USER/Documents/GitHub/Camera_vs_LIDAR/PART_A/version2_dbsacan_yolo/yolov3.weights")
    p.add_argument("--yolo_names", default="C:/Users/USER/Documents/GitHub/Camera_vs_LIDAR/PART_A/version2_dbsacan_yolo/coco.names")
    p.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold")
    p.add_argument("--nms_thresh", type=float, default=0.4, help="YOLO NMS threshold")
    return p.parse_args()

# ──────────────────────── YOLO DETECTION ──────────────────────── #

def load_yolo(cfg_path: str, weights_path: str, names_path: str):
    if not Path(cfg_path).is_file() or not Path(weights_path).is_file():
        sys.exit("YOLO cfg / weights not found. Point the script to valid files.")
    net = cv2.dnn.readNet(weights_path, cfg_path)
    with open(names_path, "r") as f:
        classes = [ln.strip() for ln in f.readlines()]
    layer_names = net.getLayerNames()
    out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, classes, out_layers


def detect_obstacles(image: np.ndarray, net, out_layers, conf_th: float, nms_th: float):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(out_layers)

    boxes, confidences, class_ids = [], [], []
    for out in outputs:
        for det in out:
            scores = det[5:]
            c_id = int(np.argmax(scores))
            conf = float(scores[c_id])
            if conf < conf_th:
                continue
            cx, cy, bw, bh = det[:4]
            x = int((cx - bw / 2) * w)
            y = int((cy - bh / 2) * h)
            bw = int(bw * w)
            bh = int(bh * h)
            boxes.append([x, y, bw, bh])
            confidences.append(conf)
            class_ids.append(c_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_th, nms_th).flatten() if boxes else []
    boxes = [boxes[i] for i in idxs]
    confidences = [confidences[i] for i in idxs]
    class_ids = [class_ids[i] for i in idxs]
    return boxes, confidences, class_ids


def boxes_to_mask(boxes, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for (x, y, w, h) in boxes:
        cv2.rectangle(mask, (max(x, 0), max(y, 0)), (min(x + w, shape[1]-1), min(y + h, shape[0]-1)), 255, thickness=-1)
    return mask

# ──────────────────── KITTI STEREO / RANSAC ──────────────────── #

# Calibration parsing helpers 

def load_kitti_calibration(calib_file):
    fx, tx = None, None
    with open(calib_file, "r") as f:
        for ln in f:
            if ln.startswith("P2:"):
                fx = float(ln.split()[1])
            elif ln.startswith("P3:"):
                tx = float(ln.split()[4])
    if fx is None or tx is None:
        raise RuntimeError("Could not parse fx/tx from calib file")
    baseline = -tx / fx
    return fx, baseline

def parse_kitti_calib(calib_file):
    with open(calib_file, "r") as f:
        lines = f.readlines()
    P2 = np.array([float(v) for v in lines[2].split()[1:]]).reshape(3, 4)
    P3 = np.array([float(v) for v in lines[3].split()[1:]]).reshape(3, 4)
    fx, cx, cy = P2[0, 0], P2[0, 2], P2[1, 2]
    Tx = -(P3[0, 3] - P2[0, 3]) / fx
    Q = np.array([[1, 0, 0, -cx],
                  [0, 1, 0, -cy],
                  [0, 0, 0, fx],
                  [0, 0, -1/Tx, 0]], dtype=np.float32)
    return Q

def compute_disparity(l_img, r_img, min_disp=0, num_disp=256, block=3):
    matcher = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=block,
                                   P1=8*3*block**2,
                                   P2=32*3*block**2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=2,
                                   preFilterCap=63,
                                   mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = matcher.compute(l_img, r_img).astype(np.float32) / 16.0
    disp[disp <= 0.0] = np.nan
    return disp

def disparity_to_pointcloud(disp, rgb, Q, obstacle_mask):
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB) / 255.0

    valid = (~np.isnan(disp)) & (obstacle_mask == 0)
    pts, cols = points[valid], colors[valid]
    pixels = np.stack(np.nonzero(valid), axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd, pixels

def segment_plane(pcd, dist=0.015, ransac_n=3, iters=1_000):
    _, inliers = pcd.segment_plane(distance_threshold=dist, ransac_n=ransac_n, num_iterations=iters)
    return np.array(inliers, dtype=int)

def refine_mask(mask, k=5, iters=2):
    'applies morphological closing and opening to refine a binary mask'
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=iters)
    return opened

def keep_largest(mask):
    'keeps the largest connected component in a binary mask'
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask
    largest = max(cnts, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [largest], -1, 255, thickness=cv2.FILLED)
    return out

def draw_boxes(image, boxes, labels):
    out = image.copy()
    for ((x, y, w, h), lbl) in zip(boxes, labels):
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(out, lbl, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return out

def overlay(image, road_mask, boxes, labels):
    out = image.copy()
    # Draw road mask (blue, alpha)
    blue = np.zeros_like(out)
    blue[:, :] = (255, 0, 0)
    mask3 = road_mask.astype(bool)[:, :, None]
    out = np.where(mask3, cv2.addWeighted(blue, 0.6, out, 0.4, 0), out)
    # Draw obstacle boxes (red)
    for ((x, y, w, h), lbl) in zip(boxes, labels):
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(out, lbl, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return out

def connect_largest_hulls(mask):
    mask_hull = np.zeros_like(mask)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        print("[WARN] nothing detected"); return

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    if len(cnts) >= 2 and cv2.contourArea(cnts[1]) > 0.45 * cv2.contourArea(cnts[0]):
        hull1= cv2.convexHull(cnts[0])
        hull2 = cv2.convexHull(cnts[1])
        cv2.fillPoly(mask_hull, [hull1], 255)
        cv2.fillPoly(mask_hull, [hull2], 255)
        road_mask= mask_hull
        #all_pts = np.vstack([cnts[0], cnts[1]])
    else:
        all_pts = cnts[0]
        hull1 = cv2.convexHull(all_pts) 
        cv2.fillPoly(mask_hull, [hull1], 255)
        road_mask= mask_hull
        print("not merged")
    return road_mask,hull1
########## lane detection

def split_lanes(image, road_hull_mask, hull, show=False,outpath=None,imgpath=None):
    print("detecting lanes")
    h,w, = image.shape[:2]
    #convert to grayscale and blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)    #binary image 255 for edge, 0 otherwise
    #show
    #cv2.imshow("edges",edges)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #edges = cv2.dilate(edges, kernel, iterations=1)
    #Region of Interest: Create a mask for the region of interest
    mask =road_hull_mask #masked_edges=edges
    #dilate the mask to make it bigger
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=2)
    cv2.fillPoly(mask, [hull], 255) #Inside the polygon: pixel value = 255 Outside:0
    masked_edges = cv2.bitwise_and(edges, mask)
    # Hough transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 360, threshold=40, minLineLength=h//4, maxLineGap=h//5)
    
    #mask center of road
    coords = np.column_stack(np.where(road_hull_mask > 0)) #πιστρέφει (y, x) θέσεις των μη μηδενικών pixels.
    if coords.size > 0:
        mask_center_x = int(np.mean(coords[:, 1]))
    else:
        mask_center_x = w // 2  # fallback αν η μάσκα είναι άδεια
    #Group lines
    distance_thresh=w*0.07 #before 30
    slope_thresh=0.1
    final_lines = []
    min_dist=abs(w//2-mask_center_x )#initial min distance
    best_line = [w//2, h, w//2, h//2] #default line in the middle of the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope1 = (y2 - y1) / (x2 - x1 + 1e-5)
        if abs(slope1) > 0.5:
            center_x=int((x1+x2)/2)
            dist = abs(center_x - mask_center_x)
            if dist < min_dist:
                min_dist = dist
                best_line = line[0] 
    left_mask = np.zeros_like(road_hull_mask)
    right_mask = np.zeros_like(road_hull_mask)
    x1, y1, x2, y2 = best_line
    center_line = [(x1, y1), (x2, y2)]
    road_mask = road_hull_mask
    # road_mask: binary μάσκα του δρόμου
    coords = np.column_stack(np.where(road_mask > 0))  # (y, x)

    # best line
    (x1, y1), (x2, y2) = center_line

    # μάσκες εξόδου
    left_mask = np.zeros_like(road_mask)
    right_mask = np.zeros_like(road_mask)

    for y, x in coords:
        # Cross product για να δεις αν το σημείο (x, y) είναι αριστερά ή δεξιά
        d = (x2 - x1)*(y - y1) - (y2 - y1)*(x - x1)
        if d < 0:
            left_mask[y, x] = 255
        else:
            right_mask[y, x] = 255
    overlay = image.copy()
    overlay[left_mask == 255] = (255, 0, 255)  # magenta
    overlay[right_mask == 255] = (255, 0, 0)   # blue
    blended = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
    i=0
    if outpath:
        os.makedirs(outpath, exist_ok=True)  # <-- απαραίτητο

        if imgpath:
            img_name = os.path.basename(imgpath).split('.')[0]
        else:
            img_name = f'image_{i}'
            i += 1

        path = os.path.join(outpath, img_name + '_lanes.png')
        success = cv2.imwrite(path, blended)
        if success:
            print(f"[INFO] Saved result to {path}")
        else:
            print(f"[ERROR] Failed to save image to {path}")
    if show:
        #old logic that didnt always show results to test many pictures stored and see the results later
        cv2.line(blended, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.imshow("lane split semi-transparent", blended)
        cv2.waitKey(0)
        cv2.destroyAllWindows()        
    return blended
# ─────────────────────────── Main ──────────────────────────── #

def process_frame(idx: str, args):
    l_path = Path(args.left_dir) / f"{idx}.png"
    r_path = Path(args.right_dir) / f"{idx}.png"
    c_path = Path(args.calib_dir) / f"{idx}.txt"
    print(f"lpath={l_path} rpath={r_path} cpath={c_path}")

    if not l_path.is_file():
        sys.exit(f"Left image {l_path} not found")

    left_full = cv2.imread(str(l_path))
    right_full = cv2.imread(str(r_path))
    if left_full is None or right_full is None:
        sys.exit("Could not load stereo pair.")

    # 1. Run YOLO on full left image
    boxes, confs, cids = detect_obstacles(left_full, yolo_net, yolo_layers, args.conf, args.nms_thresh)
    labels = [f"{yolo_classes[c]}:{conf:.2f}" for c, conf in zip(cids, confs)]
    obstacle_mask = boxes_to_mask(boxes, left_full.shape)

    # 2. Crop lower half for road extraction 
    h = left_full.shape[0]
    crop_slice = slice(h // 2, None)
    left_crop_color = left_full[crop_slice, :]
    right_crop_color = right_full[crop_slice, :]
    left_crop_gray = cv2.cvtColor(left_crop_color, cv2.COLOR_BGR2GRAY)
    right_crop_gray = cv2.cvtColor(right_crop_color, cv2.COLOR_BGR2GRAY)
    obstacle_mask_crop = obstacle_mask[crop_slice, :]

    # 3. Disparity -> point cloud (excluding obstacle pixels)
    disp = compute_disparity(left_crop_gray, right_crop_gray)
    Q = parse_kitti_calib(str(c_path))
    pcd, pixels = disparity_to_pointcloud(disp, left_crop_color, Q, obstacle_mask_crop)

    # 4. RANSAC plane segmentation
    inliers = segment_plane(pcd)

    # 5. Build full‑size road mask
    mask_full = np.zeros((h, left_full.shape[1]), dtype=np.uint8)
    pix_inliers = pixels[inliers]
    pix_inliers[:, 0] += h // 2  # shift back to full‑res coordinate space
    mask_full[pix_inliers[:, 0], pix_inliers[:, 1]] = 255
    '''show the mask
    #cv2.imshow("Road Mask", mask_full)'''
    
    #road_mask = keep_largest(refine_mask(mask_full)) #old version this and then step 7
    # 6. Connect largest hulls and split lanes
    road_mask,hull = connect_largest_hulls(refine_mask(mask_full))
    road_lanes_img = split_lanes(left_full, road_mask, hull ) #show the road lanes image
    
    # 7. Overlay
    #vis = overlay(left_full, road_mask, boxes, labels)
    final_img= draw_boxes(road_lanes_img, boxes, labels)
    #show road_lanes with boxes and labels

     
    # Show
    cv2.imshow("Road Lanes with Obstacles", final_img)
    # cv2.imshow("Road + Obstacles", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = get_args()
    # Load YOLO once
    yolo_net, yolo_classes, yolo_layers = load_yolo(args.yolo_cfg, args.yolo_weights, args.yolo_names)

    #  --index=all to process the entire folder
    if args.index.lower() == "all":
        for img_path in (Path(args.left_dir).glob("*.png")):  #  optional  sorted to see the cyclist first
            process_frame(img_path.stem, args)
    else:
        process_frame(args.index, args)
