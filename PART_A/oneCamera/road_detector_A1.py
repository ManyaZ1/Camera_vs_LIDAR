import cv2
import numpy as np
from collections import deque
from skimage.morphology import skeletonize
from skimage.draw import polygon2mask
import matplotlib.pyplot as plt


###### ROAD DETECTION PIPELINE ######
# This script implements a road detection pipeline using region growing and GrabCut.

# ===============================

# 1. ROI Construction:
#    - Define a trapezoid-shaped Region of Interest (ROI) to eliminate the sky.

# 2. C1 Channel Preprocessing:
''' - Compute the C1 channel: arctangent of R / max(G, B).
    - Apply CLAHE to enhance contrast.
    - C1 is robust to vegetation and blue sky.'''

# 3. Region Growing on C1:
#    - Start from 3 seed points near the bottom of the image.

# 4. Lab Channel Preprocessing:
''' - Apply Retinex enhancement to normalize illumination.
    - Convert to Lab color space and extract the b* channel.
    - Apply histogram equalization.'''

# 5. Region Growing on Lab b*:
#    - Start from the same 3 seed points.

# 6. Channel Selection:
'''    - Choose the best mask based on mean color statistics:
      - Use Lab by default.
      - Fall back to C1 if Lab region-growing yields green or reddish areas (vegetation warning).'''

# 7. Post-Processing of the Selected Mask:
#    - Morphological closing to fill small holes.
#    - Dilation to expand the mask and clean edges.
#    - Crop the mask to the ROI.

# 8. GrabCut Refinement:
#    - Use region-growing mask to initialize GrabCut:
#      - Core → sure foreground
#      - Rest → probable foreground
#    - Result: refined road segmentation.

# 9. Road Shape Extraction:
#    - Find the largest contour (or merge two largest if close in area).
#    - Compute the convex hull to obtain a clean road shape.

# Visual Pipeline:

#        __ C1 channel + CLAHE  \
#       /                        \ 
# ROI ---                         -- region growing → post-process → GrabCut → contour → convex hull
#       \__ Lab + Retinex ______/


###### LANE DETECTION PIPELINE ######

#TO DO: UPDATE LANE DETECTION! 
# fast lane detection -> works only for two lanes
# 1. Convert the image to grayscale and apply Gaussian blur.    
# 2. Use Canny edge detection to find edges in the image.
# 3. Create a mask for the region of interest (ROI) using the road hull mask.
# 4. Apply Hough Transform to detect lines in the masked edges.
# 5. Find the line closest to the center of the road                                           
#    - Use the cross product to determine if a point is left or right of the center line.
# 6. Create left and right masks based on the detected line.

# gray → blur → Canny → mask → Hough Transform → line detection → left/right masks



##################################################### ROAD DETECTION PIPELINE #######################################################

# --- geometric helpers -------------------------------------------------------
def build_trapezoid_roi(shape, top_ratio=0.55, bottom_ratio=0.98, widen=0.15):
    """Return a binary ROI mask shaped like a road-trapezoid."""
    h, w = shape
    pts = np.array([
        (int(w*0.5*(1-widen)), int(h*top_ratio)),   # TL
        (int(w*0.5*(1+widen)), int(h*top_ratio)),   # TR
        (int(w*bottom_ratio),  h-1),                # BR
        (int(w*(1-bottom_ratio)), h-1)              # BL
    ], dtype=np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask

def largest_hull(mask_bin):
    """Return largest contour & its convex hull (both np.ndarray) from a binary mask."""
    cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    big = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(big)
    return big, hull

def overlay_hull(frame_bgr, hull, color=(0,255,0), alpha=0.4):
    """Draw filled transparent hull on frame (in-place)."""
    overlay = frame_bgr.copy()
    cv2.fillPoly(overlay, [hull], color)
    cv2.addWeighted(overlay, alpha, frame_bgr, 1-alpha, 0, frame_bgr)

# utilities
def image_loader(root_path, image_name):
    'path->return image'
    image_path = root_path + image_name
    image = cv2.imread(image_path)
    # Check if the image was loaded successfully
    if image is None:
        print("Error loading image")
        exit(1)
    return image

# region growing
    
def compute_c1_channel(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) + 1e-6
    R = image_rgb[:, :, 0]
    G = image_rgb[:, :, 1]
    B = image_rgb[:, :, 2]
    max_GB = np.maximum(G, B) + 1e-6
    c1 = np.arctan(R / max_GB)
    return ((c1 - c1.min()) / (c1.max() - c1.min()) * 255).astype(np.uint8)


def region_growing(img, seeds, threshold)-> np.ndarray:
    """
    img: Input image (grayscale)
    seeds: List of seed points (x, y) to start the region growing
    threshold: Intensity difference threshold for region growing
    Returns a binary mask of the segmented region
    """
    h, w= img.shape
    segmented = np.zeros((h, w), np.uint8)  # Final mask
    visited = np.zeros((h, w), np.uint8)    # To avoid revisiting pixels

    for seed in seeds:
        queue = deque()
        queue.append(seed) # Initialize queue with seed point

        seed_value = img[seed[1], seed[0]]  # graysale
        region_mean = float(seed_value)
        region_size = 1

        while queue:
            x, y = queue.popleft()

            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]: #dfs
                nx, ny = x+dx, y+dy # Neighbor coordinates
                #horizon = int(h * 0.4) ; if 0 <= nx < w and horizon <= ny < h and visited[ny, nx] == 0:
                if 0 <= nx < w and 0 <= ny < h and visited[ny, nx] == 0: # Check bounds and if not visited
                    neighbor_value = img[ny, nx] # Grayscale value of neighbor
                    visited[ny, nx] = 1
                    region_mean= (region_mean * region_size + neighbor_value) / (region_size + 1) # Update mean

                    diff = abs(region_mean - neighbor_value)
                    if diff < threshold:
                        segmented[ny, nx] = 255
                        #visited[ny, nx] = 1
                        queue.append((nx, ny))
                        #region_mean += neighbor_value
                        region_size += 1

    return segmented

# grab cut
def grab_cut(img, initial_mask, iterations=5,strong_foreground_width_ratio=0.3)-> np.ndarray:
    """
    img: Input image (color, BGR) 
    initial_mask: Binary mask from region growing (0 and 255)
    itrations: Number of GrabCut iterations
    Returns the refined binary mask (0 background, 255 foreground)
    based on color information (GMMs) and pixel positions finds foreground using min-cut/max-flow optimization
    """
    h, w = initial_mask.shape
    img = img.copy()
    # # Initialize GrabCut mask
    #grabcut_mask = build_grabcut_mask_from_road_mask_old(initial_mask)  # default probable background

    # # # Models (empty, OpenCV needs them)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    grabcut_mask = build_grabcut_mask_from_road_mask(initial_mask, strong_foreground_width_ratio)

    # Apply GrabCut
    # Sanity check before calling GrabCut
    if not np.any((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD)):
        print("[ERROR] No foreground pixels in GrabCut mask — skipping.")
        return np.zeros_like(grabcut_mask, dtype=np.uint8)

    cv2.grabCut(img, grabcut_mask, None, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_MASK)

    # Post-process: Pixels marked as foreground (FG or probable FG) -> 255, else -> 0
    final_mask = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255, 0
    ).astype(np.uint8)

    return final_mask

#helper function to convert road mask to grabcut mask
def build_grabcut_mask_from_road_mask(road_mask: np.ndarray, strong_foreground_width_ratio=0.3) -> np.ndarray:
    """
    Converts a binary road_mask into grabcut mask:
    - center region becomes sure foreground (GC_FGD)
    - rest of white region becomes probable foreground (GC_PR_FGD)
    - black remains probable background (GC_PR_BGD)

    Parameters:
        road_mask: np.ndarray with 0 (background) and 255 (road)
        strong_foreground_width_ratio: width ratio of the sure foreground center band
    Returns:
        grabcut_mask: np.ndarray with values in {0,1,2,3}
    """
    h, w = road_mask.shape
    grabcut_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    grabcut_mask[road_mask == 255] = cv2.GC_PR_FGD

    # Define central "safe" zone to mark as sure foreground
    start_x = int(w * (0.5 - strong_foreground_width_ratio / 2))
    end_x = int(w * (0.5 + strong_foreground_width_ratio / 2))
    center_band = road_mask[:, start_x:end_x]

    # In that band, anything that was 255 becomes "sure foreground"
    grabcut_mask[:, start_x:end_x][center_band == 255] = cv2.GC_FGD

    return grabcut_mask

#post processing
def post_processingv1(road_mask: np.ndarray) -> np.ndarray:
    # Post-processing v1
    #close shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    # Then dilate separately (to expand)
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)) #SOS VERY IMPORTANT
    road_mask = cv2.dilate(road_mask, dilation_kernel, iterations=2)
    return road_mask



def retinex_enhanced(img_bgr, sigma=40,
                     low_L_thresh=110,  # below this Retinex is engaged
                     gamma_dark=1.3,    # only for dark scenes
                     alpha=0.6,         # blend factor L / R
                     debug=False):
    """
    Improved single‑scale Retinex with automatic gain.
    Returns an 8‑bit BGR image.
    --------------------------------------------------
    - `alpha` 0..1   :  weight of original Lightness.
    - `sigma`        :  Gaussian blur (illumination scale).
    - `low_L_thresh` :  scene‑brightness threshold (Lab L*).
    - `gamma_dark`   :  extra brightening if scene is dark.
    """

    # --- 0. to Lab ---------------------------
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L   = lab[:, :, 0] / 255.0                 # 0‑1

    # --- 1. illumination estimate ------------
    blurL = cv2.GaussianBlur(L, (0, 0), sigma)

    # --- 2. reflectance ----------------------
    reflect = np.log(L + 1e-4) - np.log(blurL + 1e-4)
    reflect -= reflect.mean()                 # centre on 0
    reflect = (reflect - reflect.min()) / (reflect.max() - reflect.min() + 1e-4)

    # --- 3. adaptive blending ----------------
    L_new = alpha * L + (1 - alpha) * reflect

    # optional gamma if scene is dark
    scene_L = L.mean()*255
    if scene_L < low_L_thresh:
        L_new = np.power(L_new, 1/gamma_dark)

    L_new = np.clip(L_new*255, 0, 255).astype(np.uint8)
    lab[:, :, 0] = L_new

    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)

    if debug:
        cv2.imshow("Retinex result", out)
        cv2.waitKey(0); cv2.destroyAllWindows()

    return out

def detect_road(img,imgpath=None,debug=False): 
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
        
    
    gc_mask = grab_cut(sel_im, rg_mask.copy(), iterations=15, strong_foreground_width_ratio=0.25)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gc_mask = cv2.morphologyEx(gc_mask, cv2.MORPH_CLOSE, kernel)
    gc_mask = cv2.dilate(gc_mask, kernel, iterations=3)
    gc_mask = cv2.bitwise_and(gc_mask, roi) 
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


def detect_road_lab(img):
    #print(f"[INFO] Detecting road using Lab method for {img}")
    dbg = True          

    # ---------- 0. preprocess & ROI ------------------
    h, w = img.shape[:2]
    roi  = build_trapezoid_roi((h, w), top_ratio=0.55)

    # Retinex only if scene is dark
    img_proc = retinex_enhanced(img, sigma=40)

    # ---------- 1. use b* channel (Lab) ---------------
    lab      = cv2.cvtColor(img_proc, cv2.COLOR_BGR2Lab)
    road_ch  = cv2.equalizeHist(lab[:, :, 2])           # b*

    # Compute local Otsu to pick threshold
    # (brighter asphalt => lower thresh ; dark => higher)
    # otsu_thresh, road_bin = cv2.threshold(road_ch, 0, 255,
    #                             cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # thr = int(0.25 * otsu_thresh)            # shrink a bit for tolerance
    # print(thr)
    thr = 15
    print(thr)
    seeds = [(w//4, int(h*0.95)), (w//2, int(h*0.95)), (3*w//4, int(h*0.95))]
    rg    = region_growing(road_ch, seeds, threshold=thr)
    rg    = post_processingv1(rg)
    mean_bgr = cv2.mean(img, mask=rg)[:3]  # discard alpha
    print("Average BGR in region-growing:", mean_bgr)
    if cv2.countNonZero(rg) < 400:
        rg = cv2.dilate(rg, np.ones((5,5), np.uint8), 1)

    # ---------- 2. GrabCut ----------------------------
    gc = grab_cut(img_proc, rg, iterations=15,
                  strong_foreground_width_ratio=0.25)

    k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    gc  = cv2.morphologyEx(gc, cv2.MORPH_CLOSE, k)
    gc  = cv2.dilate(gc, k, 2)
    gc  = cv2.bitwise_and(gc, roi)

    # ---------- 3. largest (or merged) contour --------
    cnts, _ = cv2.findContours(gc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        print("[WARN] nothing detected"); return

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # if len(cnts) >= 2 and cv2.contourArea(cnts[1]) > 0.40*cv2.contourArea(cnts[0]):
    #     merged = np.zeros_like(gc)
    #     cv2.drawContours(merged, cnts[:2], -1, 255, cv2.FILLED)
    #     cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #hull = cv2.convexHull(cnts[0])
    if len(cnts) >= 2 and cv2.contourArea(cnts[1]) > 0.35 * cv2.contourArea(cnts[0]):
        all_pts = np.vstack([cnts[0], cnts[1]])
    else:
        all_pts = cnts[0]
        print("not merged")
    
    hull = cv2.convexHull(all_pts)
    
    mask_hull = np.zeros_like(gc)
    cv2.fillPoly(mask_hull, [hull], 255)

    # ---------- 4. overlay ----------------------------
    out = img.copy()
    overlay_hull(out, hull, color=(255,0,0), alpha=0.35)

    if dbg:
        cv2.imshow("rg", rg)
        cv2.imshow("gc", gc)
        cv2.imshow("overlay", out)
        cv2.waitKey(0); cv2.destroyAllWindows()


def detect_road_c1(img,debug=False):
    img=img.copy()
    #img = retinex_enhanced(img)         NO     # <-- illumination fix
    # ---------- stage 0 : ROI so sky never enters ----------------------------
    h, w = img.shape[:2]
    roi = build_trapezoid_roi((h, w), top_ratio=0.55)
    
    # ---------- stage 1 : c1 + CLAHE inside ROI ------------------------------
    c1 = compute_c1_channel(img)
    c1_roi = cv2.bitwise_and(c1, roi)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    c1_enh = clahe.apply(c1_roi)
    # Define seeds (e.g., bottom center area)
    # ---------- stage 2 : region growing -------------------------------------
    seeds = [(w//4, int(h*0.95)), (w//2, int(h*0.95)), (3*w//4, int(h*0.95))]
    rg_mask = region_growing(c1_enh, seeds, threshold=15)
    rg_mask = post_processingv1(rg_mask)
    mean_bgr = cv2.mean(img, mask=rg_mask)[:3]  # discard alpha
    if debug:
        print("Average BGR in region-growing by c1:", mean_bgr)
     # ---------- stage 3 : GrabCut refinement ---------------------------------
    gc_mask = grab_cut(img, rg_mask, iterations=10, strong_foreground_width_ratio=0.2)
    # inflate to remove small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gc_mask = cv2.morphologyEx(gc_mask, cv2.MORPH_CLOSE, kernel)
    gc_mask = cv2.dilate(gc_mask, kernel, iterations=3)
    gc_mask = cv2.bitwise_and(gc_mask, roi)  # Keep only inside ROI

    # ---------- stage 4 : morphology → largest hull --------------------------
    big_cnt, hull = largest_hull(gc_mask)
    if hull is None:
        print('[WARN] nothing detected')
        return
    
    road_hull_mask = np.zeros_like(gc_mask)
    cv2.fillPoly(road_hull_mask, [hull], 255)

    # ---------- stage 5 : final transparent overlay --------------------------
    out = img.copy()
    overlay_hull(out, hull, color=(255, 0, 0), alpha=0.35)

    # ---------- show ---------------------------------------------------------
    if debug:
        cv2.imshow('Region Growing', rg_mask)
        cv2.imshow('GrabCut mask', gc_mask)
        cv2.imshow('Road hull overlay', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return out, road_hull_mask, hull, gc_mask, mean_bgr


####################################### LANE DETECTION PIPELINE #######################################
'''only for clearly two lanes unfortunately'''
def split_lanes(image, road_hull_mask, hull, gc_mask,outpath=None,imgpath=None):
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
    best_line = None
    #best_line = [w//2, h, w//2, h//2] #default line in the middle of the image
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
    if best_line is None:
        print("[ERROR] No suitable line found.")
        return image
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
        cv2.line(blended, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.imshow("lane split semi-transparent", blended)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if success:
            print(f"[INFO] Saved result to {path}")
        else:
            print(f"[ERROR] Failed to save image to {path}")
    else:
        #old logic that didnt always show results to test many pictures stored and see the results later
        cv2.line(blended, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.imshow("lane split semi-transparent", blended)
        cv2.waitKey(0)
        cv2.destroyAllWindows()        
    return blended
    # if outpath:
    #     img_name = imgpath.split('/')[-1].split('.')[0]
    #     path=os.path.join('road_results', img_name+ '.png')
    #     if imgpath:
    #         img_name = imgpath.split('/')[-1].split('.')[0]
    #     else:
    #         img_name = f'image_{i}'
    #         i+=1
    #     path=os.path.join(outpath, img_name+ '_lanes.png')
    #     cv2.imwrite(path, blended)
    #     print("[INFO] Saving to folder:", os.path.abspath('lane_results'))
    # else:
    #     cv2.line(blended, (x1, y1), (x2, y2), (0, 255, 0), 3)
    #     cv2.imshow("lane split semi-transparent", blended)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()


    
def split_lanes_old(image, road_hull_mask, hull, gc_mask,out=None):
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

    # Χώρισε το convex hull σε αριστερά/δεξιά σημεία σε σχέση με την best_line
    hull_array = np.array(hull).reshape(-1, 2)

    left_pts = []
    right_pts = []

    for pt in hull_array:
        px, py = pt
        # Χρήση διανυσμάτων για να δούμε αν το σημείο είναι αριστερά ή δεξιά από τη γραμμή
        d = (x2 - x1)*(py - y1) - (y2 - y1)*(px - x1)
        if d < 0:
            left_pts.append(tuple(pt))
        else:
            right_pts.append(tuple(pt))

    # Ολοκλήρωσε πολύγωνα με τη μεσαία γραμμή
    left_poly = np.array(left_pts + [center_line[1], center_line[0]], dtype=np.int32)
    right_poly = np.array(right_pts + [center_line[0], center_line[1]], dtype=np.int32)

    # Χρώμα σε μάσκες
    cv2.fillPoly(left_mask, [left_poly], 255)
    cv2.fillPoly(right_mask, [right_poly], 255)
    # Δημιουργία έγχρωμου overlay ίδιου μεγέθους με την εικόνα
    overlay = image.copy()

    # Εφαρμογή χρωμάτων στις μάσκες
    overlay[left_mask == 255] = (255, 0, 255)    # Magenta
    overlay[right_mask == 255] = (255, 0, 0)     # Blue

    # Alpha blending μεταξύ original και overlay
    alpha = 0.4  # Διαφάνεια: 0=πλήρως διαφανές, 1=πλήρως χρωματισμένο
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Προαιρετικά σχεδίασε και τη διαχωριστική γραμμή
    cv2.line(blended, center_line[0], center_line[1], (0, 255, 0), 3)

    # Εμφάνιση
    cv2.imshow("lane split semi-transparent", blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # # Δημιουργία έγχρωμης εικόνας
    # color_output = image.copy()
    # color_output[left_mask == 255] = (255, 0, 255)    # Magenta - other lane
    # color_output[right_mask == 255] = (255, 0, 0)     # Blue - my lane

    # # Προαιρετικά σχεδίασε και τη γραμμή
    # cv2.line(color_output, center_line[0], center_line[1], (0, 255, 0), 3)

    # cv2.imshow("lane split", color_output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # #draw the best line
    # cv2.line(image, (best_line[0], best_line[1]), (best_line[2], best_line[3]), (0, 255, 0), 3)
    # cv2.imshow("best line",image)
    # #show road mask
    # cv2.imshow("road mask", road_hull_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def group_lines(lines, distance_thresh=30, slope_thresh=0.1):
    final_lines = []

    for i, line1 in enumerate(lines):
        x1, y1, x2, y2 = line1[0]
        slope1 = (y2 - y1) / (x2 - x1 + 1e-5)
        print(f"line1: {x1,y1,x2,y2} slope1: {slope1}")
        merged = False

        for j, line2 in enumerate(final_lines):
            x3, y3, x4, y4 = line2
            slope2 = (y4 - y3) / (x4 - x3 + 1e-5)

            if abs(slope1 - slope2) < slope_thresh:
                dist1 = np.abs(x1 - x3) + np.abs(y1 - y3)
                dist2 = np.abs(x2 - x4) + np.abs(y2 - y4)
                if dist1 < distance_thresh or dist2 < distance_thresh:
                    # Συγχώνευση
                    new_line = [
                        (x1 + x3) // 2,
                        (y1 + y3) // 2,
                        (x2 + x4) // 2,
                        (y2 + y4) // 2
                    ]
                    final_lines[j] = new_line  # ✅ Αντικατάσταση υπάρχουσας γραμμής
                    merged = True
                    break

        if not merged:
            final_lines.append([x1, y1, x2, y2])

    return np.array(final_lines, dtype=np.int32).reshape(-1, 1, 4)    
def detect_lanes_old(image, road_hull_mask, hull, gc_mask,out=None):
    '''Blur->Canny->ROI->Hough transform->draw lines'''
    print("detecting lanes")
    h,w, = image.shape[:2]
    #convert to grayscale and blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    #canny edge detection
    edges = cv2.Canny(blur, 50, 150)    #binary image 255 for edge, 0 otherwise
    #show
    cv2.imshow("edges",edges)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #edges = cv2.dilate(edges, kernel, iterations=1)

    #Region of Interest: Create a mask for the region of interest
    mask =road_hull_mask #masked_edges=edges
    #dilate the mask to make it bigger
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=2)
    cv2.fillPoly(mask, [hull], 255) #Inside the polygon: pixel value = 255 Outside:0
    masked_edges = cv2.bitwise_and(edges, mask)
    #road_mask = road_edges_hsv(image)
    #road_roi = extract_trapezoid_from_road_mask(road_mask)
    #masked_edges = cv2.bitwise_and(edges, road_roi)
    

    # Hough transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 360, threshold=40, minLineLength=h//4, maxLineGap=h//5)

    result = image.copy()
    
    #Group lines
    tresh=w*0.07
    lines=group_lines(lines,distance_thresh=tresh)
    
    # #ignore small horizontal lines and keep most significant lines
    i=0
    dic={}
    for line in lines:
        i+=1
        x1,y1,x2,y2 = line[0]
        slope = (y2 - y1) / (x2 - x1 + 1e-5)
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        #slope = (line[0][3] - line[0][1]) / (line[0][2] - line[0][0] + 1e-5)
        if abs(slope) > 0.5 or (length)>(w//3):  # ignore small horizontal lines 
            cv2.line(result, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 255, 0), 3)
            dic[i]=length
    #         x1, y1, x2, y2 =line[0]
    #         if slope>0:
    #             rightlines.append([x1,y1,x2,y2])
    #         else:
    #             leftlines.append([x1,y1,x2,y2])
    #sort dic by length
    #dic = dict(sorted(dic.items(), key=lambda item: item[1], reverse=True))
    #take the first 3 lines
    # if len(dic)>3:
    #     dic = dict(list(dic.items())[:3])
    for line in lines:
        #line = lines[key-1]
        print(line)
        for x1, y1, x2, y2 in line:
            #cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
            if out is not None:
                cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.imshow("result",result)
    cv2.imshow("out",out)
    #show road mask
    cv2.imshow("road mask",road_hull_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return


def road_tester():
    #input_folder=r'C:/Users/USER/Documents/_CAMERA_LIDAR/image_2'
    input_folder=r'C:/Users/USER/Documents/_CAMERA_LIDAR/data_road/training/image_2'  # π.χ. './images'
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')) :
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Could not load thimage {filename}")
                continue
            print(f"Processing image: {filename}")
            start=time.time()
            #detect_road(image, img_path)
            out, road_hull_mask, hull, gc_mask=detect_road(image, imgpath=img_path, debug=True)
            split_lanes(image, road_hull_mask, hull, gc_mask,outpath='lane_results',imgpath=img_path)
            end=time.time()
            print(f"Time taken: {end-start:.2f} seconds")
# Test with a large number of images
def large_tester():
    input_folder = r'C:/Users/USER/Documents/_CAMERA_LIDAR/code/training/image_2/'  # π.χ. './images'
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')) :
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Could not load thimage {filename}")
                continue
            print(f"Processing image: {filename}")
            start=time.time()
            detect_road(image, imgpath=None, debug=True)
            #out, road_hull_mask, hull, gc_mask=detect_road(image, img_path)
            #detect_lanes(out, road_hull_mask, hull, gc_mask)
            #road_imporved_c(image)
            #detect_road_lab(image)
            #detect_road_c1(image)
            #find_edges(image)
            end=time.time()
            print(f"Time taken: {end-start:.2f} seconds")
def hardtester():
    cur_script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(cur_script_dir, 'hardtester') 
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Could not load thimage {filename}")
                continue
            print(f"Processing image: {filename}")
            start=time.time()
            out,road_hull_mask, hull, gc_mask = detect_road(image, imgpath=None, debug=False)
            end=time.time()
            print(f"Time taken: {end-start:.2f} seconds")
            cv2.imshow('Road hull overlay', out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
if __name__ == "__main__":
    import os
    import time
    x=int(input("Chose mode 1: hard tester, 2: road tester, 3: large tester: "))
    if x==1:
        print("running hard tester")
        hardtester()  # Test with hard images
    if x==2:
        print("running road tester")
        road_tester()  # KITTI dataset training road_data
    if (x)==3:
        print("running large tester")
        large_tester() #other kitti images hard

    # cur_script_dir = os.path.dirname(os.path.abspath(__file__))
    # imgpath=os.path.join(cur_script_dir, 'hardtester/um_000045.png')  # '000163.png','000221.png' , '000313.png', '000689.png', '001842.png'
    # print(f"Testing image: {imgpath}")
    # # "um_000040.png" 
    # image = cv2.imread(imgpath)
    # if image is None:
    #     print("Error loading image")
    # #     exit(1)
    #detect_road_lab(image)
    # out, road_hull_mask, hull, gc_mask=detect_road(image, imgpath=None, debug=True)
    # split_lanes(image, road_hull_mask, hull, gc_mask)
    #detect_lanes(image, road_hull_mask, hull, gc_mask,out)
    #detect_lanes_colored(image, road_hull_mask, hull, gc_mask, out=None)
    #detect_lanes_og(image)
    #detect_road_c1(image.copy(),debug=True)
    #detect_road_lab(image.copy())
    
