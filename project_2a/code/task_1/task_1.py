#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np
import glob
import os


rows = 6
cols = 9

objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

# define the path
path = os.getcwd() + '/images/task_1/'
output_path = os.getcwd() + '/output/task_1/'
# 3d point in real world space
obj_left = []
# 2d points in image plane.
img_l = []

images = glob.glob(os.path.join(path, "left_*.png"))
for img_name in images:
    img_color = cv2.imread(img_name)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(img, (rows, cols), None)
    obj_left.append(objp)
    img_l.append(corners)
    if ret == True:
        corners2 = cv2.cornerSubPix(img, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        cv2.drawChessboardCorners(img_color, (rows, cols), corners2, ret)
        cv2.imshow('img', img_color)
        cv2.imwrite(output_path + '/corners.png', img_color)

#Calibration
ret, mtx_left, dist, emp, emp = cv2.calibrateCamera(obj_left, img_l, img.shape[::-1], None, None)

# Undistortion for left_2 png
img = cv2.imread(path  + "left_2.png")

h, w = img.shape[:2]
new_mtx_left_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx_left, dist, (w,h), 0)

mapx, mapy = cv2.initUndistortRectifyMap(mtx_left, dist, None, new_mtx_left_matrix, (w,h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

if dst[0].size > 0:
    cv2.imwrite(output_path + '/left_2_undistorted.png', dst)
    undistorted_image = cv2.imread(os.path.join(output_path + '/left_2_undistorted.png'))
    cv2.imshow('undistorted image', undistorted_image)

param_path = os.getcwd() + '/parameters'

s = cv2.FileStorage('{0}/left_camera_intrinsics.xml'.format(param_path), cv2.FileStorage_WRITE)
s.write('mtx_left', mtx_left)
s.write('distCoeffs_left', dist)
s.release()

# RIght image

rows = 9
cols = 6

objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

obj_right = [] # 3d point in real world space
img_r = [] # 2d points in image plane.

images = glob.glob(os.path.join(path, "right_*.png"))

for img_name in images:
    img_color = cv2.imread(img_name)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(img, (rows, cols), None)
    obj_right.append(objp)
    img_r.append(corners)
    if ret == True:
        corners2 = cv2.cornerSubPix(img, corners, (5,5), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        cv2.drawChessboardCorners(img_color, (rows, cols), corners2, ret)
        cv2.imshow('img', img_color)

#Calibration

ret, mtx_right, dist_right, emp, emp = cv2.calibrateCamera(obj_right, img_r, img.shape[::-1], None, None)

img = cv2.imread(path + "right_2.png")

h, w = img.shape[:2]

new_mtx_right_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx_right, dist_right, (w,h), 0)

mapx, mapy = cv2.initUndistortRectifyMap(mtx_right, dist, None, new_mtx_right_matrix, (w,h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
if dst[0].size > 0:
    cv2.imwrite(output_path + '/right_2_undistorted.png', dst)
    undistorted_image = cv2.imread(output_path + '/right_2_undistorted.png')
    cv2.imshow('undistorted image', undistorted_image)

s = cv2.FileStorage('{0}/right_camera_intrinsics.xml'.format(param_path), cv2.FileStorage_WRITE)
s.write('mtx_right', mtx_right)
s.write('distCoeffs_right', dist_right)
s.release()

