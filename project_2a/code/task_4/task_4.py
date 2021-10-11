
import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
import os
from mpl_toolkits.mplot3d import axes3d, Axes3D

base_folder = os.getcwd() +'/parameters/'

s = cv2.FileStorage(base_folder + 'left_camera_intrinsics.xml', cv2.FileStorage_READ)

mtx_left = s.getNode('mtx_left').mat()
distCoeffs_left = s.getNode('distCoeffs_left').mat()
s.release()

s = cv2.FileStorage(base_folder + 'right_camera_intrinsics.xml', cv2.FileStorage_READ)
mtx_right = s.getNode('mtx_right').mat()
distCoeffs_right = s.getNode('distCoeffs_right').mat()
s.release()

s = cv2.FileStorage(base_folder + 'stereo_rectification.xml', cv2.FileStorage_READ)

R1 = s.getNode('R1').mat()
R2 = s.getNode('R2').mat()
Q = s.getNode('Q').mat()
s.release()

s = cv2.FileStorage(base_folder + 'P1.xml', cv2.FileStorage_READ)
P1 = s.getNode('P1').mat()
s = cv2.FileStorage(base_folder + 'P2.xml', cv2.FileStorage_READ)
P2 = s.getNode('P2').mat()
s.release()

img_folder = os.getcwd() + '/'


img_folder = os.getcwd() + '/'


img_l = cv2.imread(img_folder + 'images/task_3_and_4/left_4.png')

img_r= cv2.imread(img_folder + 'images/task_3_and_4/right_4.png')


height,width = img_l.shape[:2]

mapx1, mapy1 = cv2.initUndistortRectifyMap(mtx_left, distCoeffs_left, R1, mtx_left, (width,height), 5)
rectified_img_left = cv2.remap(img_l,mapx1, mapy1, cv2.INTER_LINEAR)

mapx2, mapy2 = cv2.initUndistortRectifyMap(mtx_right, distCoeffs_right,R2, mtx_right, (width,height), 5)
rectified_img_right = cv2.remap(img_r,mapx2, mapy2, cv2.INTER_LINEAR)
output_path = os.getcwd() + '/output/task_4'
cv2.imshow('rectified_img_l',rectified_img_left)
cv2.imwrite(output_path + '/rectified_img_left.png', rectified_img_left)
cv2.imshow('rectified_img_r',rectified_img_right)
cv2.imwrite(output_path + '/rectified_img_right.png', rectified_img_right)


window_size = 3
# Best parameter
left_matcher = cv2.StereoSGBM_create( minDisparity=0, numDisparities=160,blockSize=5, P1=8 * 3 * window_size ** 2,P2=32 * 3 * window_size ** 2,disp12MaxDiff=1,uniquenessRatio=15,speckleWindowSize=0,speckleRange=2, preFilterCap=63,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
#left_matcher = cv2.StereoSGBM_create(numDisparities=16, blockSize=15)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)


# FILTER Parameters
lmbda = 10000
sigma = 1.2
visual_multiplier = 0.5

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)
displ = left_matcher.compute(img_l, img_r)
dispr = right_matcher.compute(img_r, img_l)
displ = np.int16(displ)
dispr = np.int16(dispr)
filteredImg = wls_filter.filter(displ, img_l, None, dispr)

filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
filteredImg = np.uint8(filteredImg)
cv2.imshow('Disparity Map', filteredImg)
cv2.imwrite(output_path+'/Disparity.png', filteredImg)



