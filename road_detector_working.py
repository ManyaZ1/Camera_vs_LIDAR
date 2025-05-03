import cv2
import numpy as np
from collections import deque
from skimage.morphology import skeletonize
from skimage.draw import polygon2mask
import matplotlib.pyplot as plt
from curated import detect_lanes, group_lines 

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

def detect_road(img): 
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
    #mb,mg,mr = cv2.mean(img, mask=c1_rg)[:3]  # discard alpha
    
    # --- compute Lab + retinex region growing ---
    img_proc = retinex_enhanced(img, sigma=40)
    lab      = cv2.cvtColor(img_proc, cv2.COLOR_BGR2Lab)
    lab_ch = cv2.equalizeHist(lab[:, :, 2])
    lab_rg = region_growing(lab_ch, seeds, threshold=15)
    lab_rg = post_processingv1(lab_rg)
    if cv2.countNonZero(lab_rg) < 400:
        rg = cv2.dilate(lab_rg, np.ones((5,5), np.uint8), 1)
    mbl,mgl,mrl = cv2.mean(img, mask=lab_rg)[:3]  # discard alpha
    print("Average BGR in region-growing:", mbl, mgl, mrl)
    tol=5
    if mgl>mbl+tol:
        mode = 'c1'
    elif mrl>mbl+tol :
        mode='c1'
    elif mgl>mrl+tol:
        mode='c1'
    else:
        mode = 'lab'    

    # # --- choose best mask based on ROI area ---
    # def central_band_area(mask):
    #     band = np.zeros_like(mask, dtype=np.uint8)
    #     h, w = mask.shape
    #     band_h_start = int(h * 0.9)
    #     band_h_end   = h
    #     band_w_start = int(w * 0.45)
    #     band_w_end   = int(w * 0.55)
    #     band[band_h_start:band_h_end, band_w_start:band_w_end] = 255
    #     return cv2.countNonZero(cv2.bitwise_and(mask, band))
    
    # c1_area = central_band_area(c1_rg)
    # print(f"[DEBUG] C1 area: {c1_area}")
    # lab_area = central_band_area(lab_rg)
    # print(f"[DEBUG] Lab area: {lab_area}")

    # if c1_area < 300 and lab_area > c1_area:
    #     mode = 'lab'
    # elif lab_area < 300 and c1_area > lab_area:
    #     mode = 'c1'
    # else:
    #     mode = 'lab' if lab_area > c1_area else 'c1'

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

    # --- convex hull extraction ---
    big_cnt, hull = largest_hull(gc_mask)
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

    cv2.imshow('Region Growing ({} mode)'.format(mode), rg_mask)
    cv2.imshow('GrabCut mask', gc_mask)
    cv2.imshow('Road hull overlay', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def road_imporved_c(img):
    if improved_channel_selection(img) == "c1":
        print("Using C1 method")
        detect_road_c1(img)
    else:
        print("Using Lab method")
        detect_road_lab(img)




def detect_road_lab(img):
    #print(f"[INFO] Detecting road using Lab method for {img}")
    dbg = True          # set True while tuning

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
    if len(cnts) >= 2 and cv2.contourArea(cnts[1]) > 0.40 * cv2.contourArea(cnts[0]):
        all_pts = np.vstack([cnts[0], cnts[1]])
    else:
        all_pts = cnts[0]
    
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


def detect_road_c1(img):
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
    cv2.imshow('Region Growing', rg_mask)
    cv2.imshow('GrabCut mask', gc_mask)
    cv2.imshow('Road hull overlay', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Test with a large number of images
def large_tester():
    input_folder = r'C:/Users/USER/Documents/_CAMERA_LIDAR/code/training/image_2/'  # π.χ. './images'
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Could not load thimage {filename}")
                continue
            print(f"Processing image: {filename}")
            start=time.time()
            detect_road(image)
            #road_imporved_c(image)
            #detect_road_lab(image)
            #detect_road_c1(image)
            #find_edges(image)
            end=time.time()
            print(f"Time taken: {end-start:.2f} seconds")

if __name__ == "__main__":
    import os
    import time
    #large_tester()
    imgpath=os.path.join(os.getcwd(), '000689.png')  # '000163.png','000221.png' , '000313.png', '000689.png', '001842.png'
    image = cv2.imread(imgpath)
    if image is None:
        print("Error loading image")
        exit(1)
    detect_road(image)
    #detect_road_c1(image)
    #detect_road_lab(image)
    
