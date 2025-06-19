#!/usr/bin/env python3
"""
YOLO + RANSAC + GrabCut road extraction for KITTI stereo pairs.

Workflow
--------
1) YOLO detects obstacles in the left camera image and masks them so they do **not** enter the 3‑D point cloud.
2) StereoSGBM → point cloud → RANSAC plane segmentation → coarse binary road mask.
3) GrabCut takes the coarse mask + colour image and sharpens the road boundaries.
4) Overlay = road (blue translucent) + YOLO boxes (red).

# TODO
- [ ] Make more trapezoidal.
- [ ] Add lane detection.
"""

import argparse
from pathlib import Path
import sys
import os

import cv2
import numpy as np
import open3d as o3d

# ───────────────────────────── CLI ───────────────────────────── #

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Road extraction with YOLO, RANSAC, GrabCut")
    p.add_argument("--index", default="all", help="KITTI frame id (or 'all')")
    #p.add_argument("--calib_dir", default="C:/Users/Mania/Documents/KITTI/data_road/training/calib")
    #p.add_argument("--left_dir",  default="C:/Users/Mania/Documents/KITTI/data_road/training/image_2")
    #p.add_argument("--right_dir", default="C:/Users/Mania/Documents/KITTI/data_road_right/training/image_3")
    p.add_argument("--calib_dir", default="C:/Users/USER/Documents/_CAMERA_LIDAR/data_road/training/calib")
    p.add_argument("--left_dir",  default="C:/Users/USER/Documents/_CAMERA_LIDAR/data_road/training/image_2")
    p.add_argument("--right_dir", default="C:/Users/USER/Documents/_CAMERA_LIDAR/data_road_right/training/image_3")
    p.add_argument("--yolo_cfg", default="yolov3.cfg")
    p.add_argument("--yolo_weights", default="yolov3.weights")
    p.add_argument("--yolo_names", default="coco.names")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--nms_thresh", type=float, default=0.4)
    return p.parse_args()

# ───────────────────── YOLO helpers ───────────────────── #

def load_yolo(cfg: str, weights: str, names: str):
    if not Path(cfg).is_file() or not Path(weights).is_file():
        sys.exit("[FATAL] YOLO cfg/weights not found – check paths.")
    net = cv2.dnn.readNet(weights, cfg)
    with open(names) as f:
        classes = [l.strip() for l in f]
    ln = net.getLayerNames()
    out_layers = [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, classes, out_layers


def detect_obstacles(img: np.ndarray, net, layers, conf_th: float, nms_th: float):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(layers)

    boxes, scores, ids = [], [], []
    for out in outs:
        for det in out:
            s = det[5:]
            cid = int(np.argmax(s))
            score = float(s[cid])
            if score < conf_th:
                continue
            cx, cy, bw, bh = det[:4]
            x = int((cx - bw/2) * w)
            y = int((cy - bh/2) * h)
            bw = int(bw * w)
            bh = int(bh * h)
            boxes.append([x, y, bw, bh])
            scores.append(score)
            ids.append(cid)

    idxs = cv2.dnn.NMSBoxes(boxes, scores, conf_th, nms_th).flatten() if boxes else []
    return [boxes[i] for i in idxs], [scores[i] for i in idxs], [ids[i] for i in idxs]


def boxes_to_mask(boxes, shape):
    mask = np.zeros(shape[:2], np.uint8)
    for (x, y, w, h) in boxes:
        cv2.rectangle(mask, (max(0, x), max(0, y)), (min(x+w, shape[1]-1), min(y+h, shape[0]-1)), 255, -1)
    return mask

# ───────────── Stereo + point‑cloud helpers ───────────── #


def parse_kitti_calib(p):
    with open(p) as f: lines = f.readlines()
    P2 = np.fromstring(" ".join(lines[2].split()[1:]), sep=' ').reshape(3,4)
    P3 = np.fromstring(" ".join(lines[3].split()[1:]), sep=' ').reshape(3,4)
    fx, cx, cy = P2[0,0], P2[0,2], P2[1,2]
    Tx = -(P3[0,3] - P2[0,3]) / fx
    return np.array([[1,0,0,-cx],[0,1,0,-cy],[0,0,0,fx],[0,0,-1/Tx,0]], np.float32)


def compute_disparity(l_g, r_g, min_disp=0, num_disp=128, blk=5):
    sgbm = cv2.StereoSGBM_create(minDisparity=min_disp,
                                 numDisparities=num_disp,
                                 blockSize=blk,
                                 P1=8*3*blk**2,
                                 P2=32*3*blk**2,
                                 disp12MaxDiff     = 1,
                                 uniquenessRatio=5,
                                 speckleWindowSize=100,
                                 speckleRange=2,
                                 preFilterCap      = 63,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    d = sgbm.compute(l_g, r_g).astype(np.float32) / 16.0
    d[d<=0] = np.nan
    return d


def disparity_to_pcd(disp, rgb, Q, obst_mask):
    pts3 = cv2.reprojectImageTo3D(disp, Q)
    cols = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB) / 255.0
    ok = (~np.isnan(disp)) & (obst_mask == 0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3[ok])
    pcd.colors = o3d.utility.Vector3dVector(cols[ok])
    pix = np.stack(np.nonzero(ok), axis=1)
    return pcd, pix


def segment_plane(pcd, thr=0.02,ransac_n=3,iters=1000):
    _, inliers = pcd.segment_plane(distance_threshold=thr, ransac_n=ransac_n, num_iterations=iters)
    return np.array(inliers, dtype=int)
    _, inl = pcd.segment_plane(thr, 3, 1000)
    return np.array(inl, int)

# ─────────────── Morphology helpers ─────────────── #

def refine(mask, k=5, iters=2):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel, iterations=iters)
    return m

# ───────────── GrabCut refinement (not used) ───────────── #

def build_grabcut_mask_from_road_mask(road_mask: np.ndarray, strong_ratio=0.1):
    h, w = road_mask.shape
    gmask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gmask[road_mask==255] = cv2.GC_PR_FGD
    #sx = int(w*(0.5-strong_ratio/2)) 
    #ex = int(w*(0.5+strong_ratio/2))
    #band = road_mask[:, sx:ex]
    #gmask[:, sx:ex][band==255] = cv2.GC_FGD
    return gmask


def grab_cut(img_bgr: np.ndarray, coarse_mask: np.ndarray, iterations=1, strong_ratio=0.1):
    if not np.any(coarse_mask == 255):
        return coarse_mask
    h, w = coarse_mask.shape
    gmask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gmask[coarse_mask == 255] = cv2.GC_PR_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr, gmask, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    final = np.where((gmask == cv2.GC_FGD) | (gmask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return final

# ───────────────────── Road Shape Extraction ─────────────────────#
#    - Find the largest contour (or merge two largest if close in area).
def find_largest_contour(road_mask: np.ndarray):
    cnts, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        print("[WARN] nothing detected"); return
    largest = max(cnts, key=cv2.contourArea)
    mask_largest = np.zeros_like(road_mask)
    cv2.drawContours(mask_largest, [largest], -1, 255, thickness=cv2.FILLED)
    return mask_largest, largest

# Approximate the contour to a 4-point polygon and fill it      
def contour_to_quad(mask, contour, epsilon_ratio=0.02):
    # simplify → 4-vertex polygon
    peri = cv2.arcLength(contour, closed=True)
    approx = cv2.approxPolyDP(contour, epsilon_ratio * peri, closed=True)

    # if we didn’t get 4 vertices, force it with convex-hull → polyDP once more
    if len(approx) != 4:
        hull = cv2.convexHull(contour)
        approx = cv2.approxPolyDP(hull, epsilon_ratio * peri, closed=True)
        if len(approx) > 4:                        # still too many
            approx = approx[:4]

    quad_mask = np.zeros_like(mask)
    cv2.fillPoly(quad_mask, [approx], 255)
    return quad_mask
'''usage notes: 
after mask_largest = find_largest_contour(road_mask):
quad_mask   = contour_to_quad(road_mask, mask_largest)
vis_quad    = overlay(left, quad_mask, bxs, labels)
cv2.imshow("Trapezoid road", vis_quad)
'''
# Build a “vanishing-point trapezoid” from the contour’s top and bottom edges
def trapezoid_from_contour(mask, contour):
    pts = contour.reshape(-1, 2)
    h = mask.shape[0]

    # Try to extract top and bottom bands
    top_band = pts[pts[:, 1] < h * 0.3]
    bottom_band = pts[pts[:, 1] > h * 0.7]

    # Fallback if bands are empty
    if len(top_band) < 2 or len(bottom_band) < 2:
        print("[WARN] Not enough points in bands — fallback to full contour")
        top_band = bottom_band = pts

    try:
        tl = top_band[top_band[:, 0].argmin()]
        tr = top_band[top_band[:, 0].argmax()]
        bl = bottom_band[bottom_band[:, 0].argmin()]
        br = bottom_band[bottom_band[:, 0].argmax()]
    except ValueError:
        print("[ERROR] Trapezoid could not be formed — skipping")
        return np.zeros_like(mask)

    trap = np.array([tl, tr, br, bl]).reshape(-1, 1, 2)
    trap_mask = np.zeros_like(mask)
    cv2.fillPoly(trap_mask, [trap], 255)
    return trap_mask


'''usage notes: 
after mask_largest = find_largest_contour(road_mask):
trap_mask = trapezoid_from_contour(road_mask, mask_largest)
vis_trap  = overlay(left, trap_mask, bxs, labels)
cv2.imshow("Vanishing-point trapezoid", vis_trap)
'''
# Compute the convex hull to obtain a clean road shape. (not good results)

# ───────────────────── Overlay helper ───────────────────── #

def overlay(img, road_mask, boxes, labels):
    out = img.copy()
    blue = np.zeros_like(out); blue[:]=(255,0,0)
    out = np.where(road_mask[:,:,None].astype(bool), cv2.addWeighted(blue,0.6,out,0.4,0), out)
    for (x,y,w,h), lbl in zip(boxes, labels):
        cv2.rectangle(out,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(out,lbl,(x,max(0,y-8)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    return out

# ───────────────────────── Main per‑frame ───────────────────────── #

def process(idx: str, args):
    l_img = Path(args.left_dir)/f"{idx}.png"
    r_img = Path(args.right_dir)/f"{idx}.png"
    cal   = Path(args.calib_dir)/f"{idx}.txt"
    if not l_img.exists():
        print(f"[WARN] {l_img} missing – skip"); return

    left = cv2.imread(str(l_img)); right = cv2.imread(str(r_img))
    if left is None or right is None:
        print(f"[WARN] Could not load stereo pair {idx}"); return

    # 1. YOLO
    bxs,scores,cids = detect_obstacles(left,yolo_net,yolo_layers,args.conf,args.nms_thresh)
    labels=[f"{yolo_classes[c]}:{s:.2f}" for c,s in zip(cids,scores)]
    obst_mask = boxes_to_mask(bxs, left.shape)

    # 2. Lower‑half crop for geometry
    h = left.shape[0]; crop = slice(h//2,None)
    left_c, right_c = left[crop,:], right[crop,:]
    l_gray, r_gray   = cv2.cvtColor(left_c,cv2.COLOR_BGR2GRAY), cv2.cvtColor(right_c,cv2.COLOR_BGR2GRAY)
    obst_crop        = obst_mask[crop,:]

    # 3. Disparity → PCD
    disp = compute_disparity(l_gray, r_gray)
    Q    = parse_kitti_calib(str(cal))
    pcd, pix = disparity_to_pcd(disp, left_c, Q, obst_crop)

    # 4. RANSAC plane
    inl = segment_plane(pcd,thr=0.01, ransac_n=4, iters=1000)
    road_mask = np.zeros((h, left.shape[1]), np.uint8)
    pts = pix[inl]; pts[:,0]+=h//2
    road_mask[pts[:,0], pts[:,1]] = 255
    #road_mask = refine(road_mask, k=5, iters=2)

    # 5. GrabCut refinement - Expands too much 
    #road_mask = grab_cut(left, road_mask, iterations=1, strong_ratio=0.1)

    # 5. Post‑process road mask
    road_mask = refine(road_mask, k=5, iters=2)

    # 6. Find the largest contour (or merge two largest if close in area)
    mask_largest,contour = find_largest_contour(road_mask)

    # quad
    quad_mask   = contour_to_quad(road_mask, contour)
    vis_quad    = overlay(left, quad_mask, bxs, labels)
    cv2.imshow("Trapezoid road", vis_quad)

    # vanishing-point trapezoid
    trap_mask = trapezoid_from_contour(road_mask, contour)
    vis_trap  = overlay(left, trap_mask, bxs, labels)
    cv2.imshow("Vanishing-point trapezoid", vis_trap)


    vis_largest = overlay(left, mask_largest, bxs, labels)
    cv2.imshow("RANSAC+LargestContourOnly", vis_largest)

    # 6. Overlay & show
    cv2.imshow("Road + Obstacles", overlay(left, road_mask, bxs, labels))
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()




# ───────────────────────────── Runner ───────────────────────────── #

if __name__ == "__main__":
    args = get_args()
    yolo_net, yolo_classes, yolo_layers = load_yolo(args.yolo_cfg, args.yolo_weights, args.yolo_names)

    if args.index.lower()=="all":
        for p in sorted(Path(args.left_dir).glob("*.png")):
            process(p.stem, args)
    else:
        process(args.index, args)

# add contours
# cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # all_pts = np.vstack(cnts[0])    
    # hull = cv2.convexHull(all_pts)
    
    # mask_hull = np.zeros_like(road_mask)
    # cv2.fillPoly(mask_hull, [hull], 255)
    # vis_hull     = overlay(left, mask_hull, bxs, labels)
    # cv2.imshow("RANSAC+Morph+ConvexHull", vis_hull) 
