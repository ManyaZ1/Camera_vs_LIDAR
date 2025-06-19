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

## Version 1
road-lane-obstacle detection using only computer vision and color spaces without ransac:

`road_detector_A1.py `
`obstacle_detection_cv_A2.py`

## Version 2
road-lane-obstacle detection using yolo an disparity-ransac: partA1A2.py

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



