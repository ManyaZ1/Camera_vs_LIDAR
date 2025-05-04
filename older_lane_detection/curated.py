#lane detection
import cv2
import os
import numpy as np

def group_lines(lines, distance_thresh=30, slope_thresh=0.1):
    final_lines = []

    for i, line1 in enumerate(lines):
        x1, y1, x2, y2 = line1[0]
        slope1 = (y2 - y1) / (x2 - x1 + 1e-5)
        keep = True

        for j, line2 in enumerate(final_lines):
            x3, y3, x4, y4 = line2
            slope2 = (y4 - y3) / (x4 - x3 + 1e-5)

            if abs(slope1 - slope2) < slope_thresh:
                dist = np.abs(x1 - x3) + np.abs(y1 - y3)
                if dist < distance_thresh:
                    keep = False
                    #extend line 
                    x1=min(x1,x3)
                    y1=min(y1,y3)
                    x2=max(x2,x4)
                    y2=max(y2,y4)
                    break

        if keep:
            final_lines.append([x1, y1, x2, y2])

    return np.array(final_lines, dtype=np.int32).reshape(-1, 1, 4)
def chose_most_significant_line(lines):
    if not lines:
        return None
    #max length
    max_length = 0
    
    most_significant_line = None
    for line in lines:
        
        length = np.sqrt((line[2] - line[0])**2 + (line[3] - line[1])**2)
        if length > max_length:
            max_length = length
            most_significant_line = line

    return most_significant_line
def choose_lowest_line(lines):
    if not lines:
        return None
    return max(lines, key=lambda l: max(l[1], l[3]))  # line with lowest point
def shift_line(line, dx):
    x1, y1, x2, y2 = line
    return [x1 + dx, y1, x2 + dx, y2]

def detect_lanes(image):
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
    #left_main = choose_lowest_line(leftlines)
    #right_main = choose_lowest_line(rightlines)
    #keep only the most significant lines
    
    left_main = chose_most_significant_line(leftlines)
    if not left_main:
        left_main=[int(w * 0.1), int(h), int(w * 0.45), int(0.5*h)]
    right_main = chose_most_significant_line(rightlines)
    if not right_main:
        right_main=[int(w * 0.55), int(0.5*h), int(w * 0.9), int(h)]
    cv2.line(result,left_main[0:2],left_main[2:4],(255,0,0),3)
    cv2.line(result,right_main[0:2],right_main[2:4],(255,0,0),3)
    # if left_main:
    #     x1, y1, x2, y2 = left_main
    #     cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # if right_main:
    #     x1, y1, x2, y2 = right_main
    #     cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
    # if not right_main and left_main:
    #     right_est = shift_line(left_main, 400)
    #     x1, y1, x2, y2 = right_est
    #     cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # if left_main and right_main:
    #     lx1, ly1, lx2, ly2 = left_main
    #     print(left_main)
    #     print(right_main)
    #     rx1, ry1, rx2, ry2 = right_main
    #     pts = np.array([
    #         [lx1, ly1],
    #         [lx2, ly2],
    #         [rx1, ry1],
    #         [rx2, ry2]
            
    #     ])
    #     pts = pts.reshape((-1, 1, 2))
    #     cv2.polylines(result, [pts], True, (255, 0, 0), 2)
    #     #cv2.fillPoly(result, [pts], (0, 100, 200))  # fill with bluish color
    # #draw trapezoid lines
    
    # cv2.line(result, (int(w * 0.1), int(h)), (int(w * 0.45), int(0.5*h)), (0, 0, 255), 3)
    

    #show
    cv2.imshow("result",result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def image_loader(root_path, image_name):
    'path->return image'
    image_path = root_path + image_name
    image = cv2.imread(image_path)
    # Check if the image was loaded successfully
    if image is None:
        print("Error loading image")
        exit(1)
    return image


if __name__ == "__main__":
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
            detect_lanes(image)
            





