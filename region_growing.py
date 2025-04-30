import cv2
import numpy as np
from collections import deque
from skimage.morphology import skeletonize
from skimage.draw import polygon2mask
import matplotlib.pyplot as plt
# Region growing ➔ GrabCut refinement ➔ Morphology ➔ Contour ➔ Convex Hull ➔ Midline Detection ➔ Splitting σε 2 Lanes ➔ Transparent fill
# Load image
#path='C://Users//USER//Documents//_CAMERA_LIDAR//code//000005.png'
#path='C://Users//USER//Documents//_CAMERA_LIDAR//code//000015.png'
path='C://Users//USER//Documents//_CAMERA_LIDAR//code//004592.png'


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

def grab_cut(img, initial_mask, iterations=5)-> np.ndarray:
    """
    img: Input image (color, BGR) 
    initial_mask: Binary mask from region growing (0 and 255)
    itrations: Number of GrabCut iterations
    Returns the refined binary mask (0 background, 255 foreground)
    based on color information (GMMs) and pixel positions finds foreground using min-cut/max-flow optimization
    """
    h, w = initial_mask.shape

    # # Initialize GrabCut mask
    #grabcut_mask = build_grabcut_mask_from_road_mask_old(initial_mask)  # default probable background

    # # # Models (empty, OpenCV needs them)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    grabcut_mask = build_grabcut_mask_from_road_mask(initial_mask, strong_foreground_width_ratio=0.3)

    # Apply GrabCut
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

def build_grabcut_mask_from_road_mask_old(initial_mask: np.ndarray) -> np.ndarray:
    road_mask=initial_mask.copy()
    h, w = initial_mask.shape
    grabcut_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)  # default probable background
    grabcut_mask[road_mask == 255] = cv2.GC_PR_FGD
    # ενισχυμένα σίγουρα foreground (στο κέντρο μόνο)
    horizon=int(h*0.4)
    horizon=0
    center_strip = road_mask[horizon:, w//3:2*w//3]
    grabcut_mask[:, w//3:2*w//3][center_strip == 255] = cv2.GC_FGD
    return grabcut_mask

def image_loader(root_path, image_name):
    'path->return image'
    image_path = root_path + image_name
    image = cv2.imread(image_path)
    # Check if the image was loaded successfully
    if image is None:
        print("Error loading image")
        exit(1)
    return image

def post_processingv1(road_mask: np.ndarray) -> np.ndarray:
    # Post-processing v1
    #close shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    # Then dilate separately (to expand)
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)) #SOS VERY IMPORTANT
    road_mask = cv2.dilate(road_mask, dilation_kernel, iterations=2)
    return road_mask

def detect_road(img):

    #Grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Define seeds (e.g., bottom center area)
    h, w = gray.shape
    seeds = [(w//2, int(h*0.95))]  # one seed at bottom center

    # Run region growing 
    road_mask = region_growing(gray, seeds, threshold=5)

    # Post-processing v1
    road_mask = post_processingv1(road_mask)
    # GrabCut refinement
    road_mask_grabcut = grab_cut(img, road_mask,iterations=15)

    cv2.imshow('GrabCut Refined Road', road_mask_grabcut)
    #cv2.imshow('Detected Road by region growing', road_mask)
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import os
#     names=["000015.png","007121.png","000005.png","004592.png","um_000002.png"]
#    #names=["000005.png"]
#     for name in names:
#         image=image_loader(root_path="C:/Users/USER/Documents/_CAMERA_LIDAR/code/", image_name=name)
#         detect_lanes(image)
    
    input_folder = r'C:/Users/USER/Documents/_CAMERA_LIDAR/code/training/image_2/'  # π.χ. './images'
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Could not load image {filename}")
                continue
            detect_road(image)



# # Load image
# #img = cv2.imread(path)
# #grayscale image
# #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(img, (5,5), 0)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# gray_clahe = clahe.apply(gray)
# # Define seeds (e.g., bottom center area)
# h, w = gray.shape
# seeds = [(w//2, int(h*0.95))]  # one seed at bottom center

# # Run region growing
# threshold = 5  # Try different values
# road_mask = region_growing(gray, seeds, threshold)



# # Post-processing v1
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
# road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
# # Then dilate separately (to expand)
# dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)) #SOS VERY IMPORTANT
# road_mask = cv2.dilate(road_mask, dilation_kernel, iterations=2)

# Post-processing v2
# kernel = np.ones((5, 5), np.uint8)
# temp = cv2.dilate(road_mask, kernel, iterations = 2)
# temp = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel)
# temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)

##### Post-processing v3
# #dilation
# dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
# road_mask = cv2.dilate(road_mask, dilation_kernel, iterations=2)
# #erosion
# kernel = np.ones((5, 5), np.uint8)#cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
# road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
# After road_mask is computed


# ## GrabCut refinement
# road_mask_grabcut = grab_cut(img, road_mask,iterations=15)

# cv2.imshow('GrabCut Refined Road', road_mask_grabcut)
# #cv2.imshow('Detected Road by region growing', road_mask)
# cv2.imshow('Original Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
