# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 17:05:20 2021

@author: Sonu
"""

import numpy as np 
import cv2
import glob
import pickle
import os

#Setting root directory
rootDir = os.getcwd()
#Setting original frames directory.
dir_original_L = rootDir+'\Left'
dir_original_R = rootDir+'\Right'
dir_calib_process = rootDir+'\Calib'
dir_calib_parameter = rootDir+'\Parameters'
dir_compare_frame_pairs = rootDir+'\Output\Compare_rectified_frame_pairs'
dir_rectified_L = rootDir+'\Output\Rectified_frames\L'
dir_rectified_R = rootDir+'\Output\Rectified_frames\R'
dir_disparity = rootDir+'\Output\Rectified_frames\Disparity'


#Setting frame resolution.
widthPixel = 800
heightPixel = 600

#Dimensions of inner chessboard corners
Nx = 10
Ny = 7

#Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)

#Preparing the object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((Nx*Ny,3), np.float32)
objp[:,:2] = np.mgrid[0:Ny,0:Nx].T.reshape(-1,2)

#Arrays to store object points and image points from all the images
objpoints = [] #3d point in real world space
imgpointsL = [] #2d points in left image plane
imgpointsR = [] #2d points in right image plane

#Loading original frames.
os.chdir(dir_original_L) #Changing dir to the path of left frames
imagesL = glob.glob('*.jpg') #Grabbing all jpg file names
imagesL.sort() #Sorting frame file names
os.chdir(dir_original_R) #Changing dir to the path of right frames
imagesR = glob.glob('*.jpg') 
imagesR.sort() 

#Checking if the there are pairs of stereo images
if len(imagesL) != len(imagesR):
    print('Error: the image numbers of left and right cameras must be the same!')
    exit()

n = 0
for i in range(0, len(imagesL)):
    imgL = cv2.imread(dir_original_L + '\\' + imagesL[i])
    imgR = cv2.imread(dir_original_R + '\\' + imagesR[i])
    grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
    
    #Finding the chess board corners for left camera
    retL, cornersL = cv2.findChessboardCorners(grayL,(Ny,Nx),None)
    #Finding the chess board corners for right camera
    retR, cornersR = cv2.findChessboardCorners(grayR,(Ny,Nx),None)
    
    #If corners are found, add object points, image points after refining them
    if (retL and retR) == True:
        n += 1
        print('n = {}'.format(n))
        objpoints.append(objp)
        
        cornersL2 = cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),criteria)
        imgpointsL.append(cornersL2)
        
        cornersR2 = cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),criteria)
        imgpointsR.append(cornersR2)
        
        #Drawing and displaying the corners
        imgL = cv2.drawChessboardCorners(imgL, (Ny,Nx), cornersL2, retL)
        imgR = cv2.drawChessboardCorners(imgR, (Ny,Nx), cornersR2, retR)
        imgTwin = np.hstack((imgL, imgR))
        
        #Saving frame pairs in Calib directory
        cv2.imwrite(dir_calib_process + '\\' + imagesL[i][:-6] + '_corners.jpg', imgTwin)
        
print('Calculating the camera matrix, distortion coefficients, rotation and translation vectors...')

#Calibrating left camera  and getting the distortion cofficient, camera matrix, translation and rotaional matrix
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1],None,None)
#Finding the optimal left camera matrix based on alpha free scaling parameter
new_mtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(widthPixel,heightPixel), 1)

#Calibrating right camera
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1],None,None)
#Finding the optimal right camera matrix
new_mtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,(widthPixel,heightPixel), 1)
print('Done.\n')

old_mtxL = mtxL
old_mtxR = mtxR

print('Calculating reprojection errors...')
tot_errorL = 0
tot_errorR = 0
errorL = 0
errorR = 0
#Calculating reprojection errors
for i in range(len(objpoints)):
    imgpointsL2, _ = cv2.projectPoints(objpoints[i], rvecsL[i], tvecsL[i], mtxL, distL)
    imgpointsR2, _ = cv2.projectPoints(objpoints[i], rvecsR[i], tvecsR[i], mtxR, distR)
    errorL = cv2.norm(imgpointsL[i],imgpointsL2, cv2.NORM_L2)/len(imgpointsL2)
    errorR = cv2.norm(imgpointsR[i],imgpointsR2, cv2.NORM_L2)/len(imgpointsR2)
    tot_errorL += errorL
    tot_errorR += errorR
print('mean error L: ', tot_errorL/len(objpoints))
print('mean error R: ', tot_errorR/len(objpoints))

CamParasL = {'nmtxL':new_mtxL, 'distL':distL, 'rvecsL':rvecsL, 'tvecsL':tvecsL}
CamParasR = {'nmtxR':new_mtxR, 'distR':distR, 'rvecsR':rvecsR, 'tvecsR':tvecsR}

#Saving calibration parameters
print('Saving calibration parameters...')
pickle.dump(CamParasL, open(dir_calib_parameter+'\\' + 'CamParasL.p', 'wb') )
pickle.dump(CamParasR, open(dir_calib_parameter+'\\' + 'CamParasR.p', 'wb') )
print('Done.\n')

#Loading the calibration parameters
CamParasL = pickle.load( open(dir_calib_parameter+'\\' + 'CamParasL.p', 'rb' ) )
CamParasR = pickle.load( open(dir_calib_parameter+'\\' + 'CamParasR.p', 'rb' ) )
CamParasL.keys()
CamParasR.keys()

#Restoring calibration parameters from loaded dictionary
mtxL = CamParasL['nmtxL']
distL = CamParasL['distL']
rvecsL = CamParasL['rvecsL']
tvecsL = CamParasL['tvecsL']
mtxR = CamParasR['nmtxR']
distR = CamParasR['distR']
rvecsR = CamParasR['rvecsR']
tvecsR = CamParasR['tvecsR']

#Stereo Calibration
print('Cacluating the rectify parameters for stereo cameras...')
#Rectifying the stereo camera pair and finding the camera intrincs and extrinsics between them
retval, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, rotationMatrix, translationVector, essentialMatrix, fundamentalMatrix = cv2.stereoCalibrate(objpoints,imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, (widthPixel,heightPixel), criteria, flags=cv2.CALIB_FIX_INTRINSIC)

#Both mtx and cameraMatrix are same
print('mtxL = ',mtxL)
print('cameraMatrixL = ',cameraMatrixL)
print('Both mtx and cameraMatrix are same')

#Both dist and distCoeffs are same
print('distL = ',distL)
print('distCoeffsL = ',distCoeffsL)
print('Both dist and distCoeffs are same')

# cv2.stereoRectify computes the rotation matrices for each camera based on scaling parameter that brings both image along the same plane 
#It also returns projection matrix for each camera and the diparity to depth matrix which is used in point cloud generation
#Valid ROI is returned for each camera which is bounded rectangle spawning across valid pixels
rotationMatrixL, rotationMatrixR, projectionMatrixL, projectionMatrixR, disp2depthMappingMatrix, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, (widthPixel,heightPixel), rotationMatrix, translationVector, flags=cv2.CALIB_ZERO_DISPARITY, alpha=1, newImageSize=(0,0))

#rvecs and rotationMatrix are different
print('rvecsL = ',rvecsL)
print('rotationMatrixL = ',rotationMatrixL)
print('rvecs and rotationMatrix are different')

#roi and validPixROI are different
print('roiL = ', roiL)
print('validPixROI1 = ', validPixROI1)
print('roi and validPixROI are different')

print('CALIB_ZERO_DISPARITY = ', cv2.CALIB_ZERO_DISPARITY)
print('rotationMatrixL = ')
print(rotationMatrixL)
print('rotationMatrixR = ')
print(rotationMatrixR)
print('projectionMatrixL = ')
print(projectionMatrixL)
print('projectionMatrixR = ')
print(projectionMatrixR)
print('disp2depthMappingMatrix = ')
print(disp2depthMappingMatrix)
print('validPixROI1 = ')
print(validPixROI1)
print('validPixROI2 = ')
print(validPixROI2)

StereoParams = {'rotationMatrixL':rotationMatrixL, 'rotationMatrixR ':rotationMatrixR , 'projectionMatrixL':projectionMatrixL, 'projectionMatrixR':projectionMatrixR, 'disp2depthMappingMatrix':disp2depthMappingMatrix, 'validPixROI1':validPixROI1, 'validPixROI2':validPixROI2}

#Saving Stereo parameters
print('Saving stereo parameters...')
pickle.dump(StereoParams, open(dir_calib_parameter+'\\' + 'StereoParams.p', 'wb') )
print('Done.\n')

#Incorrect undistortion using projection and optimal matrix 
#mapxL, mapyL = cv2.initUndistortRectifyMap(mtxL, distL, rotationMatrixL, projectionMatrixL, (widthPixel,heightPixel), cv2.CV_32FC1)
#mapxR, mapyR = cv2.initUndistortRectifyMap(mtxR, distR, rotationMatrixR, projectionMatrixR, (widthPixel,heightPixel), cv2.CV_32FC1)

#Correct undistortion using old and new camera intrinsic matrix, rotion matrix and distortion cofficient
mapxL, mapyL = cv2.initUndistortRectifyMap(old_mtxL, distL, rotationMatrixL, mtxL, (widthPixel,heightPixel), cv2.CV_32FC1)
mapxR, mapyR = cv2.initUndistortRectifyMap(old_mtxR, distR, rotationMatrixR, mtxR, (widthPixel,heightPixel), cv2.CV_32FC1)

for i in range(len(imagesL)):
    print('i = {}'.format(i))
    print('Rectifying {} and {}...\n'.format(imagesL[i], imagesR[i]))
    imgL = cv2.imread(dir_original_L + "\\" + imagesL[i])
    imgR = cv2.imread(dir_original_R + "\\" + imagesR[i])
    hL,  wL = imgL.shape[:2]
    hR,  wR = imgR.shape[:2]
    
    #Remapping the images to produce a geometric transformation of the image
    dstL = cv2.remap(imgL, mapxL, mapyL, cv2.INTER_LINEAR)
    dstR = cv2.remap(imgR, mapxR, mapyR, cv2.INTER_LINEAR)
    
    #Combining the frame pairs
    imgTwin = np.hstack((imgL, imgR))
    imgTwin_rect = np.hstack((dstL, dstR))
    compareImg = np.vstack((imgTwin, imgTwin_rect))
    compareImg = cv2.resize(compareImg, (1024,768))
    
    cv2.imwrite(dir_compare_frame_pairs + "\\" + imagesL[i][:-6] + '_rectified.jpg', compareImg)
    #Saving rectified frames in the rectified frames directory
    cv2.imwrite(dir_rectified_L + "\\" + imagesL[i][:-4] + '_rectified.jpg', dstL)
    cv2.imwrite(dir_rectified_R + "\\" + imagesR[i][:-4] + '_rectified.jpg', dstR)

print('Calculating dispaity from rectified images')  
for i in range(len(imagesL)):
    RectL = cv2.imread(dir_rectified_L + "\\" + imagesL[i][:-4] + '_rectified.jpg')
    RectR = cv2.imread(dir_rectified_R + "\\" + imagesR[i][:-4] + '_rectified.jpg')
    #Creating a stereoSGBM object
    stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=15)
    #Finding the disparity map between the rectified stereo image pair 
    disp = stereo.compute(RectL, RectR)
    #Normalizing the dispaity map between the range of 0-255
    norm_coeff = 255 / disp.max()
    disp = disp * norm_coeff / 255
    print('Disparity map for',imagesL[i])
    #Displaying the disparity map
    cv2.imshow("disparity", disp)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    cv2.imwrite(dir_disparity + "\\" + imagesL[i][:-4] + '_disparity.jpg', disp)
