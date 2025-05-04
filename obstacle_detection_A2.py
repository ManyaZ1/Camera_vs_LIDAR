#✅ Pipeline Ανίχνευσης Εμποδίων στον Δρόμο 
#1. Disparity Map από stereo εικόνες
'Χρησιμοποίησε cv2.StereoSGBM ή cv2.StereoBM.'

'Μετατροπή του disparity σε 3D point cloud (XYZ) με reprojectImageTo3D() ή custom Q matrix.'

#2. RANSAC για εύρεση επιπέδου του εδάφους
'Εφάρμοσε RANSAC σε 3D σημεία για να βρεις το ground plane.'

'Μοντέλο: ax + by + cz + d = 0.'

#3. Αφαίρεση του επιπέδου εδάφους
'Υπολόγισε την κάθετη απόσταση κάθε σημείου από το επίπεδο.'

'Κράτησε μόνο τα σημεία πάνω από το έδαφος (π.χ. > 0.2m).'

#4. DBSCAN clustering σε 3D
'Ομαδοποίησε τα σημεία που είναι πάνω από το έδαφος.'

'Κάθε cluster θεωρείται "εμπόδιο".'

#5. Προβολή bounding boxes στην εικόνα
'Για κάθε cluster, υπολόγισε το 2D bounding box από τα σημεία του.'

'Προέβαλέ το στην εικόνα αν θέλεις visual feedback.'