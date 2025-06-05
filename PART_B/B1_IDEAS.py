# B3

DIVIDING LEFT AND RIGHT ROUGH IDEAS
1:
'''

  /    *  |
 /         \
/    *      \
line between two stars as boundary 
'''
2:
lidar_y = Tr_velo_to_cam[1, 3]  # y-translation component
road_center_y = lidar_y  # From your KITTI calibration
left_rough = rough_points[rough_points[:, 1] < road_center_y] 
right_rough = rough_points[rough_points[:, 1] > road_center_y]