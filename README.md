# Camera vs LIDAR 3D Computational Geometry and Computer Vision Project (ECE AK_811)

## 🔧 Requirements
  
- **Python** (version 3.12.9): https://www.python.org/downloads/  
- **Git Bash** (optional): For cloning the repository easily.
```
pip install opencv-python==4.5.5.64
pip install numpy==1.26.4
```

Σύστημα ανίχνευσης δρόμου και εμποδίων με χρήση LIDAR και Stereo Camera (KITTI dataset).

### Δομή
- `PART_A/oneCamera/`: One camera road detection + Disparity Obstacle detection
- `PART_A/stereoVersion`: Stereo + YOLO pipeline (Περιέχει YOLO config + βάρη)
- `PART_B/lidar_complete.py`: LIDAR-based pipeline


### Τρέξιμο

**Προσοχή**: Τα αρχεία lidar_complete.py, camera_complete.py παίνρουν ως argumements τα dierectories του KITTI dataset και του yolov3. Τα args δίνονται είτε από command line, είτε μπορούν να γίνουν set ως default στην αρχή του αρχείου στη συνάρτηση `def get_args() -> argparse.Namespace:`

Παράδειγμα εκτέλεσης:
```bash
python lidar_complete.py --index 000000 --calib_dir=... --...
python camera_complete.py --index 000000 --video
```

**Επιλογές** (για lidar_complete.py, camera_complete.py):

`--kitti`: Διατρέχει όλο το KITTI training dataset (default επιλογή)

`--testing``: Διατρέχει όλο το KITTI testing dataset 

`--video`: Δείχνει σε βίντεο τα αποτελέσματα Live και αποθηκεύει το βίντεο στο ίδιο directory με το script

`--wall`: Ανίχνευση τοίχου με τα πειραγμένα αρχεία





