# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 17:09:45 2021

@author: Sonu
"""

import cv2
import numpy as np
import pickle
import os

rootDir = os.getcwd()
dir_stereo_parameter = rootDir+'\Parameters'

StereoParams = pickle.load(open(dir_stereo_parameter +'\\' + 'StereoParams.p', 'rb') )
disp2depthMappingMatrix = StereoParams['disp2depthMappingMatrix']

calibrated_left_image = cv2.imread(r'C:\Users\Sonu\Documents\Passion\Perception_Vision\Projects\3D Scene\Disparity map & Depth Estimation\Calibrated\Output\Rectified_frames\L\frame26_l_rectified.jpg')
calibrated_right_image = cv2.imread(r'C:\Users\Sonu\Documents\Passion\Perception_Vision\Projects\3D Scene\Disparity map & Depth Estimation\Calibrated\Output\Rectified_frames\R\frame26_r_rectified.jpg')


#Calculating the disparity between a pir of stereo images
def calc_disparity(left_image, right_image):
    window_size = 3
    min_disp = 1
    num_disp = 16*2
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        disp12MaxDiff=1,
        P1=8*3*window_size**2,
        P2=32*3*window_size**2,
    )
    disp= stereo.compute(left_image, right_image).astype(np.float32) / 16.0
    
    #Normalizing the dispaity map between the range of 0-255
    norm_coeff = 255 / disp.max()
    disp = disp * norm_coeff / 255
    cv2.imshow("disparity", disp)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    cv2.imwrite('./output.jpg', disp)    
    
    return generate_point_cloud(left_image, disp)

#Generating point cloud based on disparity map        
def generate_point_cloud(left_image, disp):
    #Point cloud
    h, w = left_image.shape[:2]
    Q = disp2depthMappingMatrix
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    # write to PLY file
    create_output('output_new.ply', out_points, out_colors)
    print('%s saved' % 'out.ply')
        
    min_disp = 16
    num_disp = 112
    return (disp - min_disp) / num_disp
            

#Function to create point cloud file
def create_output(filename, vertices, colors):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')

stereo = calc_disparity(calibrated_left_image, calibrated_right_image)

