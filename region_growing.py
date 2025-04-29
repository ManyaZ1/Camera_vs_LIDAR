import cv2
import numpy as np
from collections import deque
# Region growing ➔ GrabCut refinement ➔ Morphology ➔ Contour ➔ Convex Hull ➔ Midline Detection ➔ Splitting σε 2 Lanes ➔ Transparent fill
# Load image
path='C://Users//USER//Documents//_CAMERA_LIDAR//code//000005.png'
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





# Load image
img = cv2.imread(path)
#grayscale image
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img, (5,5), 0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_clahe = clahe.apply(gray)
# Define seeds (e.g., bottom center area)
h, w = gray.shape
seeds = [(w//2, int(h*0.95))]  # one seed at bottom center

# Run region growing
threshold = 10  # Try different values
road_mask = region_growing(gray, seeds, threshold)



# Post-processing v1
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
# road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
# # Then dilate separately (to expand)
# dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
# road_mask = cv2.dilate(road_mask, dilation_kernel, iterations=2)

# Post-processing v2
# kernel = np.ones((5, 5), np.uint8)
# temp = cv2.dilate(road_mask, kernel, iterations = 2)
# temp = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel)
# temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)

##### Post-processing v3
#dilation
dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
road_mask = cv2.dilate(road_mask, dilation_kernel, iterations=2)
#erosion
kernel = np.ones((5, 5), np.uint8)#cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Detected Road', road_mask)
cv2.imshow('Original Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

