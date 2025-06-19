# Camera vs LIDAR 3D Computational Geometry and Computer Vision Project (ECE AK_811)

## 🔧 Requirements
  
- **Python** (version 3.12.9): https://www.python.org/downloads/  
- **Git Bash** (optional): For cloning the repository easily.
```
pip install opencv-python==4.5.5.64
pip install numpy==1.26.4
```

## Current Progress

## Part A
### Questions 1 and 2
**color space approach:** road_detector_A1.py 

A2:obstacle_detection_testing_A2.py

**disparity-ransac:** yolo_ransac_grabcut.py

**TO DO**
1. make mask more trapezoidal => improve the functions
2. add lanes

## Part B
### q1,2,3:
B1B2clean.py current version, B123_length exp

**TO DO**
1. improve the sidewalk removal!
change this part =>**apply better logic** (more details in B1IDEAS.py)
**αλλαξε τα percntile?**
'''

  /    *  |
 /         \
/    *      \
line between two stars as boundary 
'''
``` 
    # Median y of road
    #road_center_y = np.median(main_road[:, 1]) if len(main_road) > 0 else 0
    road_center_y=0 # τωρα => επαρκες
    left_rough = rough_points[rough_points[:, 1] < road_center_y]
    right_rough = rough_points[rough_points[:, 1] > road_center_y]
```
2. length feature of vector

3. bonus: classify obstacles



