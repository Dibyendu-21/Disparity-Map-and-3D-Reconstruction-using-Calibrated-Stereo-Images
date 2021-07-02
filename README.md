# Disparity Map and 3D Reconstruction using Calibrated Stereo Images
This repo provides information as to how to perform stereo calibration and produce the disparity map from undistorted stereo images and finally reconstruct the 3D image.

### Stereo Image Pair
![Left image](Left/frame111_l.jpg?raw=true)
![Right image](Right/frame111_r.jpg?raw=true)

## Design Pipeline
The Design Piepline is as follows:
* Read matching sets of raw left and right stereo image pairs.
* Define real world 3D coordinates of checkerboard.
* Locate the chessboard corners in image (Image Points).
### Cheesboard Corners detected in Stereo image pair
![Detected corners](Calib/frame111_corners.jpg?raw=true)
* Calibrate the left and right camera and get the distortion cofficient, camera matrix, translation and rotaional matrix for both the cameras.
* Find the optimal camera matrix (new camera) based on alpha free scaling parameter for both the cameras.
* Calibrate the stereo camera pair using stereoCalibrate() and get the rotationMatrix and translationVector for the stereo camera. 
* Compute the rotation matrices for each camera based on free scaling parameter that brings both image along the same plane using stereoRectify() and
  get diparity to depth matrix which is used in point cloud generation.
* undistort the left and right stereo images using old and new camera intrinsic matrices, rotation matrices found in earlier step
  and distortion cofficient using initUndistortRectifyMap().
* Remap the images to produce a geometric transformation of the image.
### Rectified Left Stereo image
![Rectified Left Stereo image](Output/Rectified_frames/L/frame111_l_rectified.jpg?raw=true)
### Rectified Right Stereo image
![Rectified Right Stereo image](Output/Rectified_frames/R/frame111_r_rectified.jpg?raw=true)
### Original Stereo image pair and Rectified Stereo images
![Comparision](Output/Compare_rectified_frame_pairs/frame111_rectified.jpg?raw=true)
* Create a stereo object.
* Find the disparity map between the rectified stereo image pair.
* Normalize the dispaity map between the range of 0-255.
### Disparity Map
![Disparity Map](Output/Rectified_frames/Disparity/frame111_l_disparity.jpg?raw=true)
* Generate the 3D reconstructed image (Point Cloud file) through reprojectImageTo3D() using the disparity map and the diparity to depth matrix.
### 3D Reconstructed Image
![3D Reconstructed Image](https://drive.google.com/uc?export=view&id=1RDOZDPLox7sH_LDQ5aNdYFRs0ziz9bwL)


 
