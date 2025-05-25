#!/usr/bin/env python3
# yolo_plane_road.py
# --------------------------------------------------------------
# 1) YOLO → boxes      2) disparity → 3-D cloud
# 3) mask out YOLO     4) RANSAC plane
# 5) plane mask        6) plane-aware road_detector_A1
# --------------------------------------------------------------

import cv2
import numpy as np
from obstacle_detection_cv_A2 import parser          # ← your helper
from road_detector_A1  import (                       # we re-use functions
        grab_cut, retinex_enhanced, build_trapezoid_roi,
        compute_c1_channel, region_growing, post_processingv1,
        overlay_hull, largest_hull)

# ------------------------------------------------------------------ #
# --------------- 0.  GLOBAL CONSTANTS  -----------------------------#
YOLO_CFG   = "yolov3.cfg"
YOLO_WTS   = "yolov3.weights"
COCO_NAMES = "coco.names"
CONF_THR   = 0.50
NMS_THR    = 0.40
PLANE_TOL  = 0.08        # metres from plane to accept as road
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
# --------------- 1.  UTILITY FUNCTIONS  --------------------------- #
# ------------------------------------------------------------------ #
def load_yolo():
    """Return (net, class_names, output_layers)."""
    net = cv2.dnn.readNet(YOLO_WTS, YOLO_CFG)
    names = [l.strip() for l in open(COCO_NAMES, "r")]
    layers = net.getLayerNames()
    output_layers = [layers[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, names, output_layers


def yolo_boxes(net, output_layers, img,
               conf_thr=CONF_THR, nms_thr=NMS_THR):
    """Run YOLO and return list [[x1,y1,x2,y2], …]."""
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confs = [], []
    for out in outs:
        for det in out:
            scores = det[5:]
            cls_id = np.argmax(scores)
            conf   = scores[cls_id]
            if conf > conf_thr:
                cx, cy, bw, bh = det[:4]
                x = int(cx * w - bw * w / 2)
                y = int(cy * h - bh * h / 2)
                bw = int(bw * w)
                bh = int(bh * h)
                boxes.append([x, y, x + bw, y + bh])
                confs.append(float(conf))

    idxs = cv2.dnn.NMSBoxes(
        [b[:4] for b in boxes], confs, conf_thr, nms_thr)
    return [boxes[i] for i in idxs.flatten()]


def compute_disparity(left, right):
    """StereoSGBM disparity → float32 disp (pixels)."""
    left_g  = cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY)
    right_g = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=128, blockSize=5,
        P1=8*3*5**2, P2=32*3*5**2, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2)
    return sgbm.compute(left_g, right_g).astype(np.float32) / 16.0


def ransac_plane(points, iters=1000, thr=0.03):
    """Return (a,b,c,d), inlier_mask for Nx3 points."""
    np.random.seed(0)
    best_cnt = -1
    best_abcd = (0, 1, 0, 0)
    best_mask = None
    for _ in range(iters):
        p1, p2, p3 = points[np.random.choice(len(points), 3, False)]
        n = np.cross(p2 - p1, p3 - p1)
        if np.linalg.norm(n) < 1e-6:
            continue
        a, b, c = n
        d = -np.dot(n, p1)
        dist = np.abs((points @ n) + d) / np.linalg.norm(n)
        mask = dist < thr
        cnt  = mask.sum()
        if cnt > best_cnt:
            best_cnt, best_abcd, best_mask = cnt, (a, b, c, d), mask

    # normalise & keep normal pointing upward (b > 0)
    a, b, c, d = best_abcd
    s = np.linalg.norm([a, b, c])
    a, b, c, d = (a/s, b/s, c/s, d/s)
    if b < 0:
        a, b, c, d = -a, -b, -c, -d
    return (a, b, c, d), best_mask


def build_plane_mask(plane, pts3d, valid_mask, tol=PLANE_TOL):
    """Binary mask 255 where |distance|<tol."""
    a,b,c,d = plane
    den = np.linalg.norm([a,b,c])
    dist = np.abs(a*pts3d[...,0] + b*pts3d[...,1] +
                  c*pts3d[...,2] + d) / den
    mask = (valid_mask & (dist < tol)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    return cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, 2), k, 2)


# ------------------------------------------------------------------ #
# --------------- 2.  PLANE-AWARE ROAD DETECTOR  ------------------- #
# (Minimal edit of your detect_road_new for clarity)                 #
# ------------------------------------------------------------------ #
def detect_road_plane(img, plane_mask=None, debug=False):
    """Return overlay, road_mask (convex hull) using plane mask."""
    h, w = img.shape[:2]
    roi = build_trapezoid_roi((h, w), top_ratio=0.55)
    seeds = [(w//4, int(h*0.95)), (w//2, int(h*0.95)), (3*w//4, int(h*0.95))]
    # ---- channel 1 (c1) ----
    c1 = compute_c1_channel(img)
    c1 = cv2.equalizeHist(cv2.bitwise_and(c1, roi))
    c1_rg = post_processingv1(region_growing(c1, seeds, 15))
    # ---- channel 2 (Lab-L, retinex) ----
    img_proc = retinex_enhanced(img, sigma=40)
    lab_L = cv2.equalizeHist(cv2.cvtColor(img_proc, cv2.COLOR_BGR2Lab)[:,:,2])
    lab_rg = post_processingv1(region_growing(lab_L, seeds, 15))

    # pick best channel by simple chroma heuristic
    mb,mg,mr = cv2.mean(img, mask=c1_rg)[:3]
    mbl,mgl,mrl= cv2.mean(img, mask=lab_rg)[:3]
    rg_mask = lab_rg if mbl>mb else c1_rg
    if plane_mask is not None:
        rg_mask = cv2.bitwise_and(rg_mask, plane_mask)

    gc_mask = grab_cut(img.copy(), rg_mask, 15, 0.25)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    gc_mask = cv2.dilate(cv2.morphologyEx(gc_mask, cv2.MORPH_CLOSE, kernel), kernel, 3)
    gc_mask = cv2.bitwise_and(gc_mask, roi)
    if plane_mask is not None:
        gc_mask = cv2.bitwise_and(gc_mask, plane_mask)

    hull_mask = np.zeros_like(gc_mask)
    cnt, hull = largest_hull(gc_mask)
    if hull is None:
        return None, None
    cv2.fillPoly(hull_mask, [hull], 255)
    out = img.copy(); overlay_hull(out, hull, (255,0,0), 0.35)
    if debug:
        cv2.imshow("Road hull overlay", out); cv2.waitKey(0); cv2.destroyAllWindows()
    return out, hull_mask


# ------------------------------------------------------------------ #
# --------------- 3.  MAIN PIPELINE  --------------------------------#
# ------------------------------------------------------------------ #
def process_frame(frame_id="um_000019", debug=False):
    # 1) stereo pair & intrinsic Q
    Q, left, right = parser(frame_id)

    # 2) YOLO
    net, names, out_layers = load_yolo()
    boxes = yolo_boxes(net, out_layers, left)

    # 3) mask out boxes
    disp = compute_disparity(left, right)
    pts3d = cv2.reprojectImageTo3D(disp, Q)
    valid = disp > 0
    mask_no_yolo = np.ones_like(disp, np.uint8)
    for x1,y1,x2,y2 in boxes:
        mask_no_yolo[y1:y2, x1:x2] = 0
    mask_filtered = (valid & (mask_no_yolo>0))

    # 4) RANSAC plane
    filt_pts = pts3d[mask_filtered]
    plane, inliers = ransac_plane(filt_pts)
    print("Plane:", plane, " | inliers:", inliers.sum())

    # 5) plane mask
    plane_mask = build_plane_mask(plane, pts3d, valid)

    # 6) road detector
    overlay, road_mask = detect_road_plane(left, plane_mask, debug)
    if overlay is None:
        print("[ERROR] road detection failed")
        return
    cv2.imshow("YOLO boxes", draw_boxes(left, boxes))
    cv2.imshow("Plane mask", plane_mask)
    cv2.imshow("Road overlay", overlay)
    cv2.waitKey(0); cv2.destroyAllWindows()


def draw_boxes(img, boxes):
    out = img.copy()
    for (x1,y1,x2,y2) in boxes:
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
    return out
# ------------------------------------------------------------------ #
# -------------------------  ENTRY POINT  ---------------------------#
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    process_frame("um_000019", debug=True)
