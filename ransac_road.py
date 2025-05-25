# kitti_disparity_pointcloud_ransac.py
"""
End-to-end script for processing KITTI stereo images:
1. Compute disparity map using OpenCV StereoSGBM.
2. Convert disparity to depth and generate a colored point cloud.
3. Perform RANSAC plane segmentation (e.g., road plane) with Open3D.
4. Build a binary mask of RANSAC inliers/outliers, refine it with morphology,
   and overlay the mask on the original left image.
"""

import os
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d

# ---------------------------- Paths ---------------------------- #
calib_path = 'C:\\Users\\USER\\Documents\\_CAMERA_LIDAR\\data_road\\training\\calib'
left_img_path = 'C:\\Users\\USER\\Documents\\_CAMERA_LIDAR\\data_road\\training\\image_2'
right_img_path = 'C:\\Users\\USER\\Documents\\_CAMERA_LIDAR\\data_road_right\\training\\image_3'

# ---------------------------- Calibration utils ---------------------------- #

def load_kitti_calibration(calib_file):
    fx, tx = None, None
    with open(calib_file, "r") as f:
        for line in f:
            if line.startswith("P2:"):
                parts = line.split()
                fx = float(parts[1])
            if line.startswith("P3:"):
                parts = line.split()
                tx = float(parts[4])
    if fx is None or tx is None:
        raise ValueError("Could not parse fx or tx from calibration file")
    baseline = -tx / fx
    return fx, baseline

# ---------------------------- Disparity map -------------------------------- #

def compute_disparity(left_img, right_img, min_disp=0, num_disp=192, block_size=3):
    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = matcher.compute(left_img, right_img).astype(np.float32) / 16.0
    disparity[disparity <= 0.0] = np.nan
    return disparity

def colorize_disparity(disparity):
    disp = disparity.copy()
    disp[np.isnan(disp)] = 0  # Set NaNs to 0 for visualization
    disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
    disp_uint8 = disp_norm.astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_uint8, cv2.COLORMAP_JET)
    return disp_color

# --------------------------- Point cloud ----------------------------------- #
# This function parses the KITTI calibration file to extract the projection matrix Q
def parse_kitti_calib(calib_file):
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    
    P2 = np.array([float(val) for val in lines[2].split()[1:]]).reshape(3, 4)
    P3 = np.array([float(val) for val in lines[3].split()[1:]]).reshape(3, 4)

    fx = P2[0, 0]
    cx = P2[0, 2]
    cy = P2[1, 2]
    Tx2 = P2[0, 3]
    Tx3 = P3[0, 3]
    Tx = -(Tx3 - Tx2) / fx

    Q = np.array([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0, fx],
        [0, 0, -1 / Tx, 0]
    ], dtype=np.float32)

    return Q


def disparity_to_pointcloud(disparity, rgb, Q):
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    colors = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB) / 255.0

    mask = ~np.isnan(disparity)
    pts = points_3d[mask]
    cols = colors[mask]
    pixels = np.stack(np.nonzero(mask), axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd, pixels


# ------------------------- RANSAC segmentation ----------------------------- #

def segment_plane(pcd, distance_thresh=0.015, ransac_n=3, num_iter=1000):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_thresh,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iter)
    return plane_model, np.array(inliers, dtype=int)



# ------------------------- Mask & morphology ------------------------------- #

def inliers_to_mask(inlier_indices, pixels, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    selected_pixels = pixels[inlier_indices]
    mask[selected_pixels[:, 0], selected_pixels[:, 1]] = 255
    return mask


def refine_mask(mask, kernel_size=5, iterations=2):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return opened

def keep_largest_connected_component(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask  # empty fallback

    largest_contour = max(contours, key=cv2.contourArea)
    new_mask = np.zeros_like(mask)
    cv2.drawContours(new_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return new_mask

# ------------------------- Region growing --------------------------------- #
def region_growing(mask, image, seed=None, thresh=15):
    if seed is None:
        # Κέντρο μάζας της μεγαλύτερης περιοχής
        ys, xs = np.where(mask > 0)
        if len(ys) == 0 or len(xs) == 0:
            return np.zeros_like(mask)  # fallback για κενή μάσκα
        seed = (int(np.median(ys)), int(np.median(xs)))

    out = np.zeros_like(mask)
    h, w = mask.shape
    value = int(image[seed[0], seed[1]])
    stack = [seed]
    while stack:
        y, x = stack.pop()
        if out[y, x] == 0 and abs(int(image[y, x]) - value) < thresh:
            out[y, x] = 255
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        stack.append((ny, nx))
    return out


# ------------------------- Overlay ----------------------------------------- #

def overlay_mask(image, mask, alpha=0.6, color=(0, 0, 255)):
    overlay = image.copy()
    overlay[mask == 255] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

# ------------------------- Main routine ------------------------------------ #

def road_ransac(index = "um_000019"):
    
    left_file = os.path.join(left_img_path, f"{index}.png")
    right_file = os.path.join(right_img_path, f"{index}.png")
    calib_file = os.path.join(calib_path, f"{index}.txt")

    left_img = cv2.imread(left_file, cv2.IMREAD_COLOR)
    original_left = left_img.copy()  # keep full image for overlay
    right_img = cv2.imread(right_file, cv2.IMREAD_COLOR)

    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # Crop the lower half of the images for processing
    
    h = left_gray.shape[0]
    left_gray = left_gray[left_gray.shape[0] // 2 :, :]
    right_gray = right_gray[right_gray.shape[0] // 2 :, :]
    left_img = left_img[left_img.shape[0] // 2 :, :]  # crop original for mask overlay consistency
    # Disparity and point cloud
    fx, baseline = load_kitti_calibration(calib_file)
    disparity = compute_disparity(left_gray, right_gray)
    Q = parse_kitti_calib(calib_file)
    pcd, pixels = disparity_to_pointcloud(disparity, left_img, Q)
    
    # Shift pixel coordinates back to full image scale
    pixels[:, 0] += h // 2  # shift row indices back

    # RANSAC segmentation
    plane_model, inliers = segment_plane(pcd)

    # mask = inliers_to_mask(inliers, pixels, disparity.shape)
    # mask_refined = refine_mask(mask)
    # result = overlay_mask(left_img, mask_refined)
    # Create full-size mask
    mask_full = np.zeros((h, left_gray.shape[1]), dtype=np.uint8)
    selected_pixels = pixels[inliers]
    mask_full[selected_pixels[:, 0], selected_pixels[:, 1]] = 255

    # Morphological refinement
    mask_refined = refine_mask(mask_full, kernel_size=5, iterations=2)
    # Keep largest connected component
    mask_clean = keep_largest_connected_component(mask_refined)
    # Overlay mask on original left image
    result = overlay_mask(original_left, mask_clean)
    # Overlay on original image
    #result = overlay_mask(original_left, mask_refined)


    # Display results
    #cv2.imshow("Left Image", left_img)
    disp_color = colorize_disparity(disparity)
    cv2.imshow("Disparity Map", disp_color)
    #cv2.imshow("Disparity Map", disparity / np.nanmax(disparity))  # Normalize for display
    cv2.imshow("Mask", mask_refined)
    cv2.imshow("Overlay", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite("disparity.png", cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX))
    # cv2.imwrite("mask_raw.png", mask)
    # cv2.imwrite("mask_refined.png", mask_refined)
    # cv2.imwrite("overlay.png", result)
    # print("Done. Images saved.")



def main():
    names=[fname for fname in os.listdir(left_img_path) if fname.endswith(".png")]
    for fname in names:

        index = fname.split('.')[0]
        print(f"Processing {index}...")
        road_ransac(index)  # Change index as needed
        #road_ransac_region_growing(index)  # Uncomment to use region growing

if __name__ == "__main__":
    main()


#------failed----------------------------#

def road_ransac_region_growing(index = "um_000019"):
    
    left_file = os.path.join(left_img_path, f"{index}.png")
    right_file = os.path.join(right_img_path, f"{index}.png")
    calib_file = os.path.join(calib_path, f"{index}.txt")

    left_img = cv2.imread(left_file, cv2.IMREAD_COLOR)
    original_left = left_img.copy()  # keep full image for overlay
    right_img = cv2.imread(right_file, cv2.IMREAD_COLOR)

    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # Crop the lower half of the images for processing
    
    h = left_gray.shape[0]
    left_gray = left_gray[left_gray.shape[0] // 2 :, :]
    right_gray = right_gray[right_gray.shape[0] // 2 :, :]
    left_img = left_img[left_img.shape[0] // 2 :, :]  # crop original for mask overlay consistency
    # Disparity and point cloud
    fx, baseline = load_kitti_calibration(calib_file)
    disparity = compute_disparity(left_gray, right_gray)
    Q = parse_kitti_calib(calib_file)
    pcd, pixels = disparity_to_pointcloud(disparity, left_img, Q)
    
    # Shift pixel coordinates back to full image scale
    pixels[:, 0] += h // 2  # shift row indices back

    # RANSAC segmentation
    plane_model, inliers = segment_plane(pcd)

    # mask = inliers_to_mask(inliers, pixels, disparity.shape)
    # mask_refined = refine_mask(mask)
    # result = overlay_mask(left_img, mask_refined)
    # Create full-size mask
    mask_full = np.zeros((h, left_gray.shape[1]), dtype=np.uint8)
    selected_pixels = pixels[inliers]
    mask_full[selected_pixels[:, 0], selected_pixels[:, 1]] = 255

    # Morphological refinement
    mask_refined = refine_mask(mask_full, kernel_size=5, iterations=2)
    # Keep largest connected component
    mask_clean = keep_largest_connected_component(mask_refined)
    # Overlay mask on original left image
    # Step 1: Crop the grayscale image and corresponding mask
    gray_crop = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)  # shape = (H/2, W)
    mask_crop = mask_clean[h // 2 :, :]                     # shape = (H/2, W)

    # Step 2: Region growing on cropped image
    mask_grown_crop = region_growing(mask_crop, gray_crop, thresh=25)

    # Step 3: Reconstruct full-size mask
    final_mask = np.zeros_like(mask_clean)
    final_mask[h // 2 :, :] = mask_grown_crop

    # Step 4: Overlay
    result = overlay_mask(original_left, final_mask)


    # Display results
    #cv2.imshow("Left Image", left_img)
    disp_color = colorize_disparity(disparity)
    cv2.imshow("Disparity Map", disp_color)
    #cv2.imshow("Disparity Map", disparity / np.nanmax(disparity))  # Normalize for display
    cv2.imshow("Mask", mask_refined)
    cv2.imshow("Overlay", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite("disparity.png", cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX))
    # cv2.imwrite("mask_raw.png", mask)
    # cv2.imwrite("mask_refined.png", mask_refined)
    # cv2.imwrite("overlay.png", result)
    # print("Done. Images saved.")

def road_ransac_region_growing_old(index = "um_000019"):
    from road_detector_A1 import region_growing
    left_file = os.path.join(left_img_path, f"{index}.png")
    right_file = os.path.join(right_img_path, f"{index}.png")
    calib_file = os.path.join(calib_path, f"{index}.txt")

    left_img = cv2.imread(left_file, cv2.IMREAD_COLOR)
    original_left = left_img.copy()  # keep full image for overlay
    right_img = cv2.imread(right_file, cv2.IMREAD_COLOR)

    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # Crop the lower half of the images for processing
    
    h = left_gray.shape[0]
    left_gray = left_gray[left_gray.shape[0] // 2 :, :]
    right_gray = right_gray[right_gray.shape[0] // 2 :, :]
    left_img = left_img[left_img.shape[0] // 2 :, :]  # crop original for mask overlay consistency
    # Disparity and point cloud
    fx, baseline = load_kitti_calibration(calib_file)
    disparity = compute_disparity(left_gray, right_gray)
    Q = parse_kitti_calib(calib_file)
    pcd, pixels = disparity_to_pointcloud(disparity, left_img, Q)
    
    # Shift pixel coordinates back to full image scale
    pixels[:, 0] += h // 2  # shift row indices back

    # RANSAC segmentation
    plane_model, inliers = segment_plane(pcd)

    # mask = inliers_to_mask(inliers, pixels, disparity.shape)
    # mask_refined = refine_mask(mask)
    # result = overlay_mask(left_img, mask_refined)
    # Create full-size mask
    mask_full = np.zeros((h, left_gray.shape[1]), dtype=np.uint8)
    selected_pixels = pixels[inliers]
    mask_full[selected_pixels[:, 0], selected_pixels[:, 1]] = 255

    # Morphological refinement
    mask_refined = refine_mask(mask_full, kernel_size=5, iterations=2)
    # Keep largest connected component
    mask_clean = keep_largest_connected_component(mask_refined)
    # Region growing to expand the mask
    left_gray_full = cv2.cvtColor(original_left, cv2.COLOR_BGR2GRAY)
    seeds = np.column_stack(np.where(mask_clean == 255))
    seeds = [tuple(p[::-1]) for p in seeds] 
    mask_region = region_growing(left_gray_full, seeds, threshold=6)


    # Overlay mask on original left image
    result = overlay_mask(original_left, mask_region)

    #result = overlay_mask(original_left, mask_clean)
    #result = overlay_mask(original_left, mask_refined)


    # Display results
    #cv2.imshow("Left Image", left_img)
    disp_color = colorize_disparity(disparity)
    cv2.imshow("Disparity Map", disp_color)
    #cv2.imshow("Disparity Map", disparity / np.nanmax(disparity))  # Normalize for display
    cv2.imshow("Mask", mask_refined)
    cv2.imshow("Overlay", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite("disparity.png", cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX))
    # cv2.imwrite("mask_raw.png", mask)
    # cv2.imwrite("mask_refined.png", mask_refined)
    # cv2.imwrite("overlay.png", result)
    # print("Done. Images saved.")