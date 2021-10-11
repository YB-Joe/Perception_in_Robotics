import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# This parameters are coming from the instruction
mtx_left = np.array([[423.27381306, 0, 341.34626532],
                     [0, 421.27401756, 269.28542111],
                     [0, 0, 1]])


dist_left = np.array([-0.43394157423038077, 0.26707717557547866, -0.00031144347020293427, 0.0005638938101488364, -0.10970452266148858])

mtx_right = np.array([[420.91160482, 0, 352.16135589],
                      [0, 418.72245958, 264.50726699],
                      [0, 0, 1]])

dist_right = np.array([-0.4145817681176909, 0.19961273246897668, -0.00014832091141656534, -0.0013686760437966467, -0.05113584625015141])

'''
Undistort the loaded image and extract 2D-TO-2D point correspondences
'''
img_1 = cv2.imread("../../images//task_5/left_0.png")
map = cv2.initUndistortRectifyMap(mtx_left, dist_left, None, None,(640,480),cv2.CV_32FC1)
undist_image1 = cv2.remap(img_1, map[0], map[1], cv2.INTER_LINEAR)
cv2.imshow("Undistorted image left_0", undist_image1)
cv2.imwrite('../../output/task_5/Undistorted_left.png', undist_image1)
## Undistorting left_1 image
img_2 = cv2.imread("../../images/task_5/right_0.png")
map2 = cv2.initUndistortRectifyMap(mtx_left, dist_left, None, None,(640,480),cv2.CV_32FC1)
undist_image2 = cv2.remap(img_2, map2[0], map2[1], cv2.INTER_LINEAR)
cv2.imshow("Undistorted image right_0", undist_image2)
cv2.imwrite('../../output/task_5/Undistorted_right.png', undist_image2)
# step2: find the corner points with undistorted image and draw the corners on image
grayscaled = cv2.cvtColor(undist_image1, cv2.COLOR_BGR2GRAY)
ret, corner1 = cv2.findChessboardCorners(grayscaled, (9,6),None)
corner1 = np.float32(corner1).reshape(-1,2)
left_corner = cv2.drawChessboardCorners(undist_image1, (9,6), corner1, ret)
cv2.imshow("Corners on left_0", left_corner)
cv2.imwrite('../../output/task_5/Cornersleft.png', left_corner)
grayscaled2 = cv2.cvtColor(undist_image2, cv2.COLOR_BGR2GRAY)
ret, corner2 = cv2.findChessboardCorners(grayscaled2, (9,6),None)
corner2 = np.float32(corner2).reshape(-1,2)
corner_image_right = cv2.drawChessboardCorners(undist_image2, (9,6), corner2, ret)
cv2.imshow("Corners on right_0", corner_image_right)
cv2.imwrite('../../output/task_5/Cornersright.png', corner_image_right)
# use the OpenCV library function "findHomography()" to calculate the homography.
# Many of you have doubts about the coordinates on the world plane. Here are my points.
points = \
 np.array([[300, 800,   0],
 [310, 800,   0],
 [320, 800,   0],
 [330, 800,   0],
 [340, 800,   0],
 [350, 800,   0],
 [360, 800,   0],
 [370, 800,   0],
 [380, 800,   0],
 [300, 810,   0],
 [310, 810,   0],
 [320, 810,   0],
 [330, 810,   0],
 [340, 810,   0],
 [350, 810,   0],
 [360, 810,   0],
 [370, 810,   0],
 [380, 810,   0],
 [300, 820,   0],
 [310, 820,   0],
 [320, 820,   0],
 [330, 820,   0],
 [340, 820,   0],
 [350, 820,   0],
 [360, 820,   0],
 [370, 820,   0],
 [380, 820,   0],
 [300, 830,   0],
 [310, 830,   0],
 [320, 830,   0],
 [330, 830,   0],
 [340, 830,   0],
 [350, 830,   0],
 [360, 830,   0],
 [370, 830,   0],
 [380, 830,   0],
 [300, 840,   0],
 [310, 840,   0],
 [320, 840,   0],
 [330, 840,   0],
 [340, 840,   0],
 [350, 840,   0],
 [360, 840,   0],
 [370, 840,   0],
 [380, 840,   0],
 [300, 850,   0],
 [310, 850,   0],
 [320, 850,   0],
 [330, 850,   0],
 [340, 850,   0],
 [350, 850,   0],
 [360, 850,   0],
 [370, 850,   0],
 [380, 850,   0]], dtype=np.float32)

matrix1, mask = cv2.findHomography(corner1, points, cv2.RANSAC, 5.0)
matrix2, mask = cv2.findHomography(corner2, points, cv2.RANSAC, 5.0)
print('Homography matrix for left image')
print(matrix1)
print('Homography matrix for right image')
print(matrix2)

'''
use the OpenCV library function "warpPerspective()" with the homography you calculated in the previous step to reconstruct the 2D plane, as shown in Figure 3. 
'''
perspective1 = cv2.warpPerspective(left_corner, matrix1, (960, 1280))
perspective2 = cv2.warpPerspective(corner_image_right, matrix2, (960, 1280))
width = img_1.shape[0]
height = img_1.shape[1]
cv2.imshow("perspective", cv2.resize(perspective1, (width, height), interpolation = cv2.INTER_AREA))
cv2.imshow("perspective", cv2.resize(perspective2, (width, height), interpolation = cv2.INTER_AREA))
cv2.imwrite('../../output/task_5/perspective1.png', perspective1)
cv2.imwrite('../../output/task_5/perspective2.png', perspective2)
fig, (ax1,ax2) = plt.subplots(1, 2)
ax1.imshow(cv2.cvtColor(perspective1 , cv2.COLOR_BGR2RGB), interpolation=None)
ax2.imshow(cv2.cvtColor(perspective2 , cv2.COLOR_BGR2RGB), interpolation=None)

ax2.set_xlabel('\n 2D world reconstructed from the left_0 & right_0 image')
plt.show()

