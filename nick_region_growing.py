import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from time import time
from skimage.segmentation import random_walker

# Type Annotations
from typing import List, Tuple
from cv2.typing import MatLike

def region_growing(img: MatLike, seed: Tuple[int, int], threshold: int) -> np.ndarray:
    '''
    Γυρνάει την μάσκα (0 & 255) της περιοχής βάση του seed που δώσαμε.

    img  -> Εικόνα (αναγκαστικά με γκρι απόχρωση)
    seed -> Το αρχικό pixel που θα ξεκινήσει την αναζήτηση
    '''
    (height, width) = img.shape
    segmented = np.zeros((height, width), np.uint8)
    visited = np.zeros_like(segmented, dtype = bool) # Όλα false αρχικά!
    
    region_intensity = img[seed]
    region_pixels = [seed]
    neighbors = [
        (-1, -1), (-1, 0), (-1, +1),
        ( 0, -1),          ( 0, +1),
        (+1, -1), (+1, 0), (+1, +1)
    ]

    (count, sum_intensity) = (0, 0)
    while region_pixels:
        (x, y) = region_pixels.pop()
        if visited[x, y]:
            continue;

        visited[x, y] = True
        intensity = int(img[x, y])

        # Δυναμική ενημέρωση του μέσου όρου!
        sum_intensity += intensity
        count += 1
        region_intensity = sum_intensity / count

        if abs(intensity - region_intensity) <= threshold:
            segmented[x, y] = 255 # Άσπρο pixel

            for (dx, dy) in neighbors:
                (xn, yn) = (x + dx, y + dy)
                if (0 <= xn < height) and (0 <= yn < width) and (not visited[xn, yn]):
                    region_pixels.append((xn, yn))

    return segmented;

def apply_grabcut(image: MatLike, mask: np.ndarray) -> np.ndarray:
    '''
    Γυρνάει την τελική μάσκα μετά το GrabCut.

    image -> Έγχρωμη εικόνα (BGR format)
    mask  -> Το αποτέλεσμα του region growing
    '''
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        image,
        mask,
        None,
        bgModel,
        fgModel,
        15, # Όσα περισσότερα iterations, τόσο καλύτερα!
        cv2.GC_INIT_WITH_MASK
    )

    # Final binary mask: 1 or 3 = foreground
    grabcut_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype('uint8')
    
    return grabcut_mask;

def apply_grabcut_fast(image: MatLike,
                       mask: np.ndarray,
                       downscale: float = 0.5) -> np.ndarray:
    '''
    Γρηγορότερο GrabCut μέσω χρήσης μικρότερης εικόνας!
    
    image -> Έγχρωμη εικόνα (BGR format)
    mask  -> Το αποτέλεσμα του region growing
    downscale -> Πόσο να μικρύνει η εικόνα (π.χ. 0.5 = 50%)
    '''
    (height, width) = image.shape[:2] # [:2] γιατί περιέχει και τα 3 κανάλια (BGR)!
    small_size = (int(width * downscale), int(height * downscale))

    small_image = cv2.resize(image, small_size, interpolation = cv2.INTER_LINEAR)
    small_mask  = cv2.resize(mask, small_size, interpolation = cv2.INTER_NEAREST)
    # cv2.INTER_NEAREST -> Ώστε να παραμένουν οι τιμές της μάσκας 0 & 255!

    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        small_image,
        small_mask,
        None,
        bgModel,
        fgModel,
        15,
        cv2.GC_INIT_WITH_MASK
    )

    result_small = np.where((small_mask == 1) | (small_mask == 3), 255, 0).astype('uint8')
    result_big = cv2.resize(result_small, (width, height), interpolation=cv2.INTER_NEAREST)

    return result_big;

def apply_random_walker(image: MatLike, mask: np.ndarray) -> np.ndarray:
    '''
    Εφαρμογή του Random Walker segmentation!

    * image -> Εικόνα (αναγκαστικά με γκρι απόχρωση)
    * Η είσοδος mask πρέπει να έχει:
      1 -> background
      2 -> foreground
      0 -> unknown
    '''
    labels = random_walker(image, mask, beta = 20000, mode = 'bf', tol = 1e-4)
    # Όσο μεγαλύτερο το beta, τόσο καλύτερα!

    result = np.where(labels == 2, 255, 0).astype('uint8') # Όπου labels == 2 -> δρόμος!

    return result;

def my_road_detection(image: MatLike,
                      image_color: MatLike,
                      method: str = 'grabcut') -> Tuple[np.ndarray, np.ndarray]:
    '''
    Συνδυάζει region growing και grabcut για να βρει την μάσκα του δρόμου.
    Παράλληλα, εφαρμόζει και κάποιες μορφολογικές μετασχηματίσεις για να καθαρίσει την εικόνα!
    Τέλος, επιστρέφει την τελική μάσκα και το μεγαλύτερο contour που βρήκε.
    '''
    (height, width) = image.shape

    seed_points = (height - 1, width//2) # Το μόνο σίγουρο seed...
    if method == 'grabcut': # Βάση δοκιμών!
        threshold = 5
    elif method == 'grabcut_fast':
        threshold = 10
    elif method == 'random_walker':
        threshold = 15
    seg_result = region_growing(image, seed = seed_points, threshold = threshold)

    ' https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html '
    kernel = np.ones((5, 5), np.uint8)
    temp = cv2.dilate(seg_result, kernel, iterations = 2)
    temp = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel)
    temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)

    if method == 'grabcut':
        temp = np.where(temp == 255, cv2.GC_FGD, cv2.GC_PR_BGD).astype('uint8')
        # Εάν seg_result == 255 -> True, τότε GC_FGD, αλλιώς GC_PR_BGD (πιθανό background)
        mask = apply_grabcut(image_color, temp)
    elif method == 'grabcut_fast':
        temp = np.where(temp == 255, cv2.GC_FGD, cv2.GC_PR_BGD).astype('uint8')
        mask = apply_grabcut_fast(image_color, temp, downscale = 0.4)
    elif method == 'random_walker':
        seeds = np.ones_like(temp, dtype = np.uint8) # Όλα background (1)
        seeds[temp == 255] = 2 # Σίγουρα δρόμος, λόγω region growing!
        dilated = cv2.dilate((seeds == 2).astype(np.uint8), kernel, iterations = 6)
        seeds[(dilated == 1) & (seeds != 2)] = 0 # Unknown area!
        mask = apply_random_walker(image, seeds)
    else:
        raise ValueError(f"Άγνωστη μέθοδος: {method}");

    ' https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html '
    (contours, _) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key = cv2.contourArea) # Μεγαλύτερο contour σε επιφάνεια!
    final_mask = np.zeros_like(mask)
    cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness = cv2.FILLED)

    return (final_mask, largest_contour);

def find_mask_vertices(largest_contour: np.ndarray) -> List[Tuple[int, int]]:
    '''
    Βρίσκει τις κορυφές της περιμέτρου της μάσκας!
    Επιστρέφει λίστα με σημεία [(x, y), (x, y), ...]
    '''
    peri = cv2.arcLength(largest_contour, True) # Υπολογισμός περιμέτρου
    epsilon = 0.01 * peri # Το 0.01 δίνει συνήθως πάνω από 4 σημεία!

    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    vertices = [tuple(pt[0]) for pt in approx]

    return vertices;

def CH_graham_scan(points: np.ndarray) -> np.ndarray: # Χορηγία του Lab 2!
    if len(points) < 3:
        raise ValueError("No convex hull can be defined with less than 3 points!");
    
    def orientation(p, q, r):
        return np.cross(q - p, r - q);

    # Initial step - sorting according to angle
    p0_idx = np.argmin(points[:, 1])
    p0 = points[p0_idx]

    # Calculate the polar angle of each point with respect to p0
    angles = np.arctan2(points[:, 1] - p0[1], points[:, 0] - p0[0])

    # Sort the points by angle
    sorted_idx = np.argsort(angles)
    sorted_points = points[sorted_idx]

    stack = [0, 1] # Η λίστα πρχ δείκτες! Σίγουρα το 0 και 1, σκέψου το γιατί!
    ids = np.arange(2, sorted_points.shape[0])

    # Iterating over the rest of the points
    for id in ids:
        # Pop elements from the stack until graham condition is satisfied
        while (len(stack) > 1) and not orientation(sorted_points[stack[-2]],
                                                   sorted_points[stack[-1]],
                                                   sorted_points[id]) > 0:
            stack.pop()

        stack.append(id) # Append current point

    return sorted_points[np.array(stack)];

def lane_separation(image: MatLike,
                    road_mask: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    '''
    image -> Πρέπει να είναι γκρι εικόνα!
    road_mask -> Μάσκα με 0 & 255
    Επιστρέφει 2 σημεία που ορίζουν την ευθεία που χωρίζει τα 2 lanes!
    '''
    edges = cv2.Canny(image, 50, 150)
    edges = cv2.bitwise_and(edges, road_mask)
    lines = cv2.HoughLines(
        edges,
        rho = 1,
        theta = np.pi/180,
        threshold = 60
    )

    # Βρίσκουμε τη γραμμή πιο κοντά στο κέντρο της μάσκας!
    coords = np.column_stack(np.where(road_mask > 0))
    mask_center_x = int(np.mean(coords[:, 1]))

    best_line = None
    best_score = float('inf')
    for line in lines:
        (rho, theta) = line[0]

        '''
        Κάνει την εύρεση της γραμμής σχεδόν αδύνατη...
        angle_deg = np.degrees(theta)
        if not (80 <= angle_deg <= 100):
            continue; # Μόνο σχεδόν κατακόρυφες γραμμές!
        '''
        
        (a, b) = (np.cos(theta), np.sin(theta))
        (x0, y0) = (a * rho, b * rho)
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        center_x = (x1 + x2) // 2 # Δεν δουλεύει πάντα, αλλά είναι γρήγορο κριτήριο!
        dist = abs(center_x - mask_center_x)
        if dist < best_score:
            best_score = dist
            best_line = ((x1, y1), (x2, y2))

    return best_line;

def split_convex_hull(
    convex_hull: List[Tuple[int, int]],
    middle_line: Tuple[Tuple[int, int], Tuple[int, int]]
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    '''
    Χωρίζει το convex hull σε 2 lanes, με βάση τη μέση γραμμή που δώσαμε!
    '''
    intersections = []
    for i in range(len(convex_hull)):
        A = convex_hull[i]
        B = convex_hull[(i+1) % len(convex_hull)] # Κυκλικά

        if intersect(A, B, middle_line[0], middle_line[1]):
            intersections.append((A, B))
        
    p_intersection1 = compute_intersection(
        intersections[0][0], intersections[0][1],
        middle_line[0], middle_line[1]
    )
    p_intersection2 = compute_intersection(
        intersections[1][0], intersections[1][1],
        middle_line[0], middle_line[1]
    )

    # Σε ποια σημεία του convex hull συμβαίνουν τα intersections
    idx1 = find_index_of_point(convex_hull, intersections[0][0])
    idx2 = find_index_of_point(convex_hull, intersections[1][0])

    # lane1 = από idx1 μέχρι idx2 (κυκλικά) και προσθέτουμε τα intersections
    lane1 = []
    lane1.append(p_intersection1)
    i = idx1
    while i != idx2:
        lane1.append(tuple(convex_hull[i]))
        i = (i + 1) % len(convex_hull)
    lane1.append(tuple(convex_hull[idx2]))
    lane1.append(p_intersection2)

    # lane2 = υπόλοιπο
    lane2 = []
    lane2.append(p_intersection2)
    i = idx2
    while i != idx1:
        lane2.append(tuple(convex_hull[i]))
        i = (i + 1) % len(convex_hull)
    lane2.append(tuple(convex_hull[idx1]))
    lane2.append(p_intersection1)

    return (lane1, lane2);

def my_road_is(
    image: MatLike,
    image_color: MatLike,
    method: str = 'grabcut'
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], np.ndarray]:
    '''
    Επιστρέφει τα 2 lanes (2 convex hulls) που χωρίζουν τον δρόμο σε 2 λωρίδες
    και το convex hull του δρόμου!
    '''
    (road_mask, largest_contour) = my_road_detection(image, image_color, method)
    corners = find_mask_vertices(largest_contour)
    convex_hull = CH_graham_scan(np.array(corners))
    middle_white_line = lane_separation(image, road_mask) # Ή κίτρινη...
    (lane1, lane2) = split_convex_hull(convex_hull, middle_white_line)

    return (lane1, lane2, convex_hull);

# --- Helpers ---

def _ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
    ''' Επιστρέφει True αν τα σημεία A, B, C είναι counter-clockwise. '''
    return ((C[1] - A[1]) * (B[0] - A[0])) > ((B[1] - A[1]) * (C[0] - A[0]));

def intersect(A: Tuple[float, float], B: Tuple[float, float],
              C: Tuple[float, float], D: Tuple[float, float]) -> bool:
    ''' Επιστρέφει True αν το τμήμα AB τέμνει το τμήμα CD. '''
    return (_ccw(A, C, D) != _ccw(B, C, D)) and (_ccw(A, B, C) != _ccw(A, B, D));

def compute_intersection(A: Tuple[float, float], B: Tuple[float, float],
                         C: Tuple[float, float], D: Tuple[float, float]) -> Tuple[float, float]:
    '''
    Βρίσκει το σημείο τομής των ευθειών AB και CD.
    '''
    A = np.array(A, dtype = float)
    B = np.array(B, dtype = float)
    C = np.array(C, dtype = float)
    D = np.array(D, dtype = float)

    s = np.vstack([A, B, C, D])        
    h = np.hstack((s, np.ones((4, 1))))

    l1 = np.cross(h[0], h[1])
    l2 = np.cross(h[2], h[3])

    (x, y, z) = np.cross(l1, l2)

    return (x/z, y/z);

def draw_lanes(image_color: MatLike,
               lane1: List[Tuple[float, float]],
               lane2: List[Tuple[float, float]]) -> None:
    # Ζωγραφικήηη
    overlay = image_color.copy()

    lane1 = np.array(lane1, dtype = np.int32)
    lane2 = np.array(lane2, dtype = np.int32)

    cv2.fillPoly(overlay, [lane1], color = (255, 0, 0)) # Μπλε γεμάτο
    cv2.fillPoly(overlay, [lane2], color = (0, 255, 0)) # Πράσινο γεμάτο

    alpha = 0.4 # Συνδυασμός με το αρχικό image
    cv2.addWeighted(overlay, alpha, image_color, 1 - alpha, 0, dst = image_color)

    # Περίγραμμα για να φαίνεται καλύτερα!
    cv2.polylines(image_color, [lane1], isClosed = True, color = (0, 0, 0), thickness = 2)
    cv2.polylines(image_color, [lane2], isClosed = True, color = (0, 0, 0), thickness = 2)

    return;

def draw_road_convex_hull(image_color: MatLike, convex_hull: np.ndarray) -> None:
    # Ζωγραφικήηη
    overlay = image_color.copy()

    convex_hull = np.array(convex_hull, dtype = np.int32)
    cv2.fillPoly(overlay, [convex_hull], color = (0, 0, 255)) # Κόκκινο γεμάτο

    alpha = 0.4 # Συνδυασμός με το αρχικό image
    cv2.addWeighted(overlay, alpha, image_color, 1 - alpha, 0, dst = image_color)

    # Περίγραμμα για να φαίνεται καλύτερα!
    cv2.polylines(
        image_color,
        [convex_hull],
        isClosed = True,
        color = (0, 0, 0),
        thickness = 2
    )

    return;

def find_index_of_point(convex_hull: np.ndarray, point: Tuple[int, int]) -> int:
    '''
    Επιστρέφει το index του convex_hull στο οποίο αντιστοιχεί το point!
    '''
    matches = np.all(np.isclose(convex_hull, point), axis = 1)
    indices = np.where(matches)[0]
    
    return indices[0];

def polygon_area(vertices: List[Tuple[float, float]]) -> float:
    '''
    Υπολογίζει το εμβαδόν ενός κλειστού πολυγώνου με τον τύπο του Shoelace.

    https://en.wikipedia.org/wiki/Shoelace_formula
    '''
    n = len(vertices)
    area = 0.

    for i in range(n):
        (x0, y0) = vertices[i]
        (x1, y1) = vertices[(i+1) % n] # Κυκλικό!

        area += (x0 * y1) - (x1 * y0)

    return abs(area) / 2.;

def compute_c1_channel(image_color: np.ndarray) -> np.ndarray:
    '''
    Υπολογισμός του c1 καναλιού για την έγχρωμη εικόνα (BGR).
    Το αποτέλεσμα είναι float32 και normalized μεταξύ [0, 1]!

    c1(x, y) = arctan(R / max(G, B))

    Paper:
    Random-Walker Monocular Road Detection in
    Adverse Conditions Using Automated
    Spatiotemporal Seed Selection
    '''
    # BGR -> RGB
    image_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB).astype(np.float32) + 1e-6
    R = image_rgb[:, :, 0]
    G = image_rgb[:, :, 1]
    B = image_rgb[:, :, 2]

    max_GB = np.maximum(G, B) + 1e-6  # Για αποφυγή διαίρεσης με το 0
    c1 = np.arctan(R / max_GB)

    c1_normalized = (c1 - c1.min()) / (c1.max() - c1.min()) # Κανονικοποίηση στο [0, 1]

    return c1_normalized.astype(np.float32);

# --- Παράδειγμα χρήσης ---

def main():
    base_dir = os.path.dirname(__file__)

    for i in range(0, 10):
        # temp = os.path.join(
        #     base_dir,
        #     'data_road',
        #     'testing',
        #     'image_2',
        #     f'um_00000{i}.png'
        # )
        temp=os.path.join(base_dir,"000015.png")#+"//000015.png"
        img_file = os.path.abspath(temp)

        # Φορτώνουμε γκρι και έγχρωμη εικόνα!
        image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        image_color = cv2.imread(img_file, cv2.IMREAD_COLOR) # Σε BGR format!
        image_c1_channel = compute_c1_channel(image_color)

        if image is None or image_color is None:
            print(f"Η εικόνα {img_file} δεν βρέθηκε!")
            continue;
        
        try:
            start = time()
            (lane1, lane2, convex_hull) = my_road_is(
                image,
                image_color,
                method = 'grabcut'
            )
            print(f"Διάρκεια εκτέλεσης: {time() - start:.2f} sec")

            (lane1_area, lane2_area) = (polygon_area(lane1), polygon_area(lane2))
            temp = lane1_area/lane2_area if lane1_area > lane2_area else lane2_area/lane1_area

            # Ζωγραφικήηη
            if temp < 10:
                draw_lanes(image_color, lane1, lane2)
            else:
                print('Δεν βρέθηκε αξιοπιστη διαχωριστική γραμμή! Απλός σχεδιασμός δρόμου...')
                draw_road_convex_hull(image_color, convex_hull)
        except Exception as e:
            print(e)
            print(f"Σφάλμα στην εικόνα {img_file}!")
            continue;

        plt.figure(figsize = (10, 6))

        # Πρέπει να γίνει μετατροπή σε RGB format, για να εμφανιστεί σωστά στο matplotlib!
        plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))

        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close()

    return;

if __name__ == "__main__":
    main()
    # path='C://Users//USER//Documents//_CAMERA_LIDAR//code//000015.png'
    # #image = cv2.imread(path)
    # img_file = os.path.abspath(path)
    # # Φορτώνουμε γκρι και έγχρωμη εικόνα!
    # image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    # image_color = cv2.imread(img_file, cv2.IMREAD_COLOR) # Σε BGR format!
    # image_c1_channel = compute_c1_channel(image_color)

    # if image is None or image_color is None:
    #     print(f"Η εικόνα {img_file} δεν βρέθηκε!")
    
    # try:
    #     start = time()
    #     (lane1, lane2, convex_hull) = my_road_is(image, image_color, method='grabcut')

    #     print(f"Διάρκεια εκτέλεσης: {time() - start:.2f} sec")

    #     (lane1_area, lane2_area) = (polygon_area(lane1), polygon_area(lane2))
    #     temp = lane1_area/lane2_area if lane1_area > lane2_area else lane2_area/lane1_area

    #     # Ζωγραφικήηη
    #     if temp < 2.2:
    #         draw_lanes(image_color, lane1, lane2)
    #         plt.figure(figsize=(10, 6))
    #         plt.imshow(cv2.cvtColor(image_color.copy(), cv2.COLOR_BGR2RGB))
    #         plt.title("Detected Road & Lanes")
    #         plt.axis('off')
    #         plt.tight_layout()
    #         plt.show()

    #     else:
    #         print('Δεν βρέθηκε αξιοπιστη διαχωριστική γραμμή! Απλός σχεδιασμός δρόμου...')
    #         draw_road_convex_hull(image_color, convex_hull)
    #         # Εμφάνιση έγχρωμης εικόνας με lanes επάνω της
    #         plt.figure(figsize=(10, 6))
    #         plt.imshow(cv2.cvtColor(image_color.copy(), cv2.COLOR_BGR2RGB))
    #         plt.title("Detected Road & Lanes")
    #         plt.axis('off')
    #         plt.tight_layout()
    #         plt.show()
    # except Exception as e:
    #     print(e)
    #     print(f"Σφάλμα στην εικόνα {img_file}!")

    

