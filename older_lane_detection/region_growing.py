import cv2
import numpy as np
from collections import deque
from skimage.morphology import skeletonize
from skimage.draw import polygon2mask
import matplotlib.pyplot as plt
from curated import detect_lanes, group_lines 


def region_growing_with_edges(img: np.ndarray, seed, threshold: int, edge_map: np.ndarray) -> np.ndarray:
    h, w = img.shape
    segmented = np.zeros((h, w), np.uint8)
    visited = np.zeros((h, w), np.uint8)

    queue = deque()
    queue.append(seed)
    region_mean = float(img[seed[1], seed[0]])
    region_size = 1

    while queue:
        x, y = queue.popleft()

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy

            if 0 <= nx < w and 0 <= ny < h and visited[ny, nx] == 0:
                if edge_map[ny, nx] == 255:
                    continue  # Μην περάσεις από edge

                neighbor_value = img[ny, nx]
                visited[ny, nx] = 1
                region_mean = (region_mean * region_size + neighbor_value) / (region_size + 1)

                if abs(region_mean - neighbor_value) < threshold:
                    segmented[ny, nx] = 255
                    queue.append((nx, ny))
                    region_size += 1

    return segmented

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

def grab_cut(img, initial_mask, iterations=5,strong_foreground_width_ratio=0.3)-> np.ndarray:
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


# --- Morphology cleanup after GrabCut ---
def morphology_cleanup(road_mask: np.ndarray) -> np.ndarray:
    # 1. Remove small blobs (noise)
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
    final_road_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
    # # 2. Keep only the largest contour (likely the road)
    # contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if contours:
    #     largest_contour = max(contours, key=cv2.contourArea)
    #     filtered_mask = np.zeros_like(cleaned_mask)
    #     cv2.drawContours(filtered_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    # else:
    #     filtered_mask = cleaned_mask.copy()

    # # 3. Skeletonize the road shape
    # skeleton = skeletonize((filtered_mask > 0).astype(np.uint8))

    # # 4. Expand region around skeleton
    # skeleton_coords = np.column_stack(np.where(skeleton > 0))
    # dilated_skeleton = np.zeros_like(skeleton, dtype=np.uint8)
    # for y, x in skeleton_coords:
    #     cv2.circle(dilated_skeleton, (x, y), radius=10, color=1, thickness=-1)

    # # Final refined road mask
    # final_road_mask = (dilated_skeleton * 255).astype(np.uint8)
    return final_road_mask
    
def compute_c1_channel(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) + 1e-6
    R = image_rgb[:, :, 0]
    G = image_rgb[:, :, 1]
    B = image_rgb[:, :, 2]
    max_GB = np.maximum(G, B) + 1e-6
    c1 = np.arctan(R / max_GB)
    return ((c1 - c1.min()) / (c1.max() - c1.min()) * 255).astype(np.uint8)
def filter_by_area(mask, min_area_ratio=0.1, max_area_ratio=0.7):
    h, w = mask.shape
    total_area = h * w
    detected_area = np.sum(mask == 255)
    ratio = detected_area / total_area

    print(f"[INFO] Detected road area ratio: {ratio:.3f}")
    if ratio < min_area_ratio or ratio > max_area_ratio:
        print("[WARNING] Rejected mask due to abnormal size.")
        return np.zeros_like(mask)
    return mask

def detect_road(img):

    #Grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = compute_c1_channel(img)
    #gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75) #makes it worse

    # Define seeds (e.g., bottom center area)
    h, w = gray.shape
    seeds = [(w//2, int(h*0.95))]  # one seed at bottom center

    # Run region growing 
    #road_mask = region_growing(gray, seeds, threshold=15)
    #cv2.imshow('Region Growing Road Mask', road_mask)
    h, w = gray.shape
    seeds = [
    (int(w * 0.25), int(h * 0.95)),
    (int(w * 0.5),  int(h * 0.95)),
    (int(w * 0.75), int(h * 0.95))
    ]
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #gray = clahe.apply(gray)
    road_mask = region_growing(gray, seeds, threshold=15)
    #road_mask = filter_by_area(road_mask)

    cv2.imshow('Region Growing Road Mask', road_mask)
    imgmask = cv2.bitwise_and(img, img, mask=road_mask)
    #find_edges(imgmask)
    # Post-processing v1
    road_mask = post_processingv1(road_mask)
    # GrabCut refinement
    road_mask_grabcut = grab_cut(img, road_mask,iterations=15)

    # Morphology cleanup after GrabCut
    #final_road_mask = morphology_cleanup(road_mask_grabcut)
    # Show result
    #cv2.imshow('Final Refined Road Mask', final_road_mask)

    cv2.imshow('GrabCut Refined Road', road_mask_grabcut)
    #cv2.imshow('Detected Road by region growing', road_mask)
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_roadv1(img):
    'problem:sometimes sky is included in the mask'
    c1 = compute_c1_channel(img)  # float32 in [0, 1]
    c1_uint8 = (c1 * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_c1 = clahe.apply(c1_uint8)

    #gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75) #makes it worse

    # Define seeds (e.g., bottom center area)
    h, w,_ = img.shape
    #seeds = [(w//2, int(h*0.95))]  # one seed at bottom center
    seeds = [
    (int(w * 0.25), int(h * 0.95)),
    (int(w * 0.5),  int(h * 0.95)),
    (int(w * 0.75), int(h * 0.95))
    ]
    
    road_mask = region_growing(enhanced_c1, seeds, threshold=15)
    #road_mask = filter_by_area(road_mask)

    cv2.imshow('Region Growing Road Mask', road_mask)
    imgmask = cv2.bitwise_and(img, img, mask=road_mask)
    #find_edges(imgmask)
    # Post-processing v1
    road_mask = post_processingv1(road_mask)
    # GrabCut refinement
    road_mask_grabcut = grab_cut(img, road_mask,iterations=15)

    # Morphology cleanup after GrabCut
    #final_road_mask = morphology_cleanup(road_mask_grabcut)
    # Show result
    #cv2.imshow('Final Refined Road Mask', final_road_mask)

    cv2.imshow('GrabCut Refined Road', road_mask_grabcut)
    #cv2.imshow('Detected Road by region growing', road_mask)
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def filter_connected_components(mask, min_area=3000):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = np.zeros_like(mask)
    for i in range(1, num_labels):  # skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            result[labels == i] = 255
    return result

def detect_roadv2(img):
    'problem:sometimes sky is included in the mask'
    c1 = compute_c1_channel(img)  # float32 in [0, 1]
    c1_uint8 = (c1 * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_c1 = clahe.apply(c1_uint8)

    #gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75) #makes it worse

    # Define seeds (e.g., bottom center area)
    h, w,_ = img.shape
    #seeds = [(w//2, int(h*0.95))]  # one seed at bottom center
    seeds = [
    (int(w * 0.25), int(h * 0.95)),
    (int(w * 0.5),  int(h * 0.95)),
    (int(w * 0.75), int(h * 0.95))
    ]
    
    road_mask = region_growing(enhanced_c1, seeds, threshold=15)
    #road_mask = filter_connected_components(road_mask, min_area=3000)


   
    #_, c1_thresh = cv2.threshold((c1 * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #non_road = (c1_thresh == 255)

    # Exclude these from region growing output
    #road_mask[non_road] = 0
    
    
    road_mask = post_processingv1(road_mask)
    cv2.imshow('Region Growing Road Mask', road_mask)
    # GrabCut refinement
    road_mask_grabcut = grab_cut(img, road_mask,iterations=15)

    # Morphology cleanup after GrabCut
    #final_road_mask = morphology_cleanup(road_mask_grabcut)
    # Show result
    #cv2.imshow('Final Refined Road Mask', final_road_mask)

    cv2.imshow('GrabCut Refined Road', road_mask_grabcut)
    #cv2.imshow('Detected Road by region growing', road_mask)
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def smart_region_growing(img, seeds, threshold):
    "try with one seed if area is too small try with 3 seeds if area too large limit it" 
    #Grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = compute_c1_channel(img)
    #gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75) #makes it worse

    # Define seeds (e.g., bottom center area)
    h, w = gray.shape
    seeds = [(w//2, int(h*0.95))]  # one seed at bottom center

    # Run region growing 
    road_mask = region_growing(gray, seeds, threshold=15)
    h, w = road_mask.shape
    total_area = h * w
    detected_area = np.sum(road_mask == 255)
    ratio = detected_area / total_area
    if ratio<0.1:
    #cv2.imshow('Region Growing Road Mask', road_mask)
        h, w = gray.shape
        seeds = [
        (int(w * 0.25), int(h * 0.95)),
        (int(w * 0.5),  int(h * 0.95)),
        (int(w * 0.75), int(h * 0.95))
        ]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    road_mask = region_growing(gray, seeds, threshold=15)
    

    return
    
def test1(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    h, w = image.shape[:2]
    roi_vertices = np.array([[   
            (w * 0.1, h),           # bottom left
        (w * 0.45, h * 0.5),    # top left
        (w * 0.55, h * 0.5),    # top right
        (w * 0.9, h)            # bottom right
        ]], dtype=np.int32)

    #ROI  Create a mask for the region of interest
    mask = np.zeros_like(edges) #masked_edges=edges
    cv2.fillPoly(mask, roi_vertices, 255) #Inside the polygon: pixel value = 255 Outside:0
    edges = cv2.bitwise_and(edges, mask)

def testSobel(image):
    angle_thresh_deg=30
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
# Sobel derivatives
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # dx
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # dy

    magnitude = np.sqrt(sobelx**2 + sobely**2)
    angle = np.arctan2(np.abs(sobely), np.abs(sobelx)) * 180 / np.pi  # angle in degrees

    # Keep only edges with angle far from horizontal (i.e., close to vertical)
    angle_mask = (angle > angle_thresh_deg) & (angle < (180 - angle_thresh_deg))

    # Combine with magnitude threshold
    edge_mask = (magnitude > 50) & angle_mask
    return edge_mask.astype(np.uint8) * 255


def find_edges(image):
    '''Blur->Canny->ROI->Hough transform->draw lines'''
    #convert to grayscale and blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #canny edge detection
    edges = cv2.Canny(blur, 50, 150)    #binary image 255 for edge, 0 otherwise
    #Region of Interest: 
    h, w = image.shape[:2]
    roi_vertices = np.array([[   
        (w * 0.1, h),           # bottom left
        (w * 0.45, h * 0.5),    # top left
        (w * 0.55, h * 0.5),    # top right
        (w * 0.9, h)            # bottom right
    ]], dtype=np.int32)
    roi_vertices = np.array([[
        (w * 0.1, h),           # bottom left
        (w * 0.1, h * 0.5),    # top left
        (w * 0.9, h * 0.5),    # top right
        (w * 0.9, h)            # bottom right
    ]], dtype=np.int32)
    #ROI  Create a mask for the region of interest
    mask = np.zeros_like(edges) #masked_edges=edges
    cv2.fillPoly(mask, roi_vertices, 255) #Inside the polygon: pixel value = 255 Outside:0
    masked_edges = cv2.bitwise_and(edges, mask)
    #road_mask = road_edges_hsv(image)
    #road_roi = extract_trapezoid_from_road_mask(road_mask)
    #masked_edges = cv2.bitwise_and(edges, road_roi)
    

    # Hough transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 360, threshold=40, minLineLength=h/4., maxLineGap=60)
    
    #Group lines
    tresh=w*0.05
    lines=group_lines(lines,tresh)
    result = image.copy()
    #ignore horizontal lines and keep most significant lines
    leftlines=[]
    rightlines=[]
    for line in lines:
        #remove horizontal lines
        slope = (line[0][3] - line[0][1]) / (line[0][2] - line[0][0] + 1e-5)
        if abs(slope) > 0.5 :  # ignore horizontal lines 
            #cv2.line(result, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 255, 0), 3)
            x1, y1, x2, y2 =line[0]
            if slope>0:
                rightlines.append([x1,y1,x2,y2])
            else:
                leftlines.append([x1,y1,x2,y2])
    # Draw all lines after filtering
    for (x1, y1, x2, y2) in leftlines + rightlines:
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Show final image
    cv2.imshow('Lines', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
            detect_roadv2(image)
            #find_edges(image)
            end=time.time()
            print(f"Time taken: {end-start:.2f} seconds")

if __name__ == "__main__":
    import os
    import time
    #large_tester()
    imgpath=os.path.join(os.getcwd(), '000221.png')  # '000163.png','000221.png'
    image = cv2.imread(imgpath)
    if image is None:
        print("Error loading image")
        exit(1)
    detect_roadv1(image)
    #detect_lanes(image)
    find_edges(image)                                                          #''''''

#     names=["000015.png","007121.png","000005.png","004592.png","um_000002.png"]
#    #names=["000005.png"]
#     for name in names:
#         image=image_loader(root_path="C:/Users/USER/Documents/_CAMERA_LIDAR/code/", image_name=name)
#         detect_lanes(image)
    
    
            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # filtered_edges = testSobel(image)
            # cv2.imshow("Only Vertical-ish Edges", filtered_edges)
            # #cv2.imshow('Road Mask with Edges', mask)
            # cv2.imshow('Edges', filtered_edges)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # #test(image)
        



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
