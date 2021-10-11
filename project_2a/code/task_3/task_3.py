import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
import os
from mpl_toolkits.mplot3d import axes3d, Axes3D

base_folder = os.getcwd() +'/parameters/'

s = cv2.FileStorage(base_folder + 'left_camera_intrinsics.xml', cv2.FileStorage_READ)

mtxl = s.getNode('mtx_left').mat()
distl = s.getNode('distCoeffs_left').mat()
s.release()

s = cv2.FileStorage(base_folder + 'right_camera_intrinsics.xml', cv2.FileStorage_READ)
mtxr = s.getNode('mtx_right').mat()
distr = s.getNode('distCoeffs_right').mat()
s.release()

s = cv2.FileStorage(base_folder + 'P1.xml', cv2.FileStorage_READ)
P1 = s.getNode('P1').mat()
s.release()
s = cv2.FileStorage(base_folder + 'P2.xml', cv2.FileStorage_READ)
P2 = s.getNode('P2').mat()
s.release()

img_folder = os.getcwd() + '/'

img_l = cv2.imread(img_folder + 'images/task_3_and_4/left_0.png')

img_r= cv2.imread(img_folder + 'images/task_3_and_4/right_0.png')


grady_scaled_left= cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)
grady_scaled_right= cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)

height,width = img_l.shape[:2]

mapx1, mapy1 = cv2.initUndistortRectifyMap(mtxl, distl, None, mtxl, (width,height), 5)
undistorted_left = cv2.remap(grady_scaled_left,mapx1, mapy1, cv2.INTER_LINEAR)

mapx2, mapy2 = cv2.initUndistortRectifyMap(mtxr, distr, None, mtxr, (width,height), 5)
undistorted_right = cv2.remap(grady_scaled_right,mapx2, mapy2, cv2.INTER_LINEAR)

cv2.imshow('img_l',undistorted_left)
cv2.imshow('img_r',undistorted_right)

orb=cv2.ORB_create()
feature_left= orb.detect(undistorted_left,None)
feature_left, feature_left1 = orb.compute(undistorted_left,feature_left)
key_point = cv2.drawKeypoints(undistorted_left,feature_left,np.array([]),(0,250,0),0)
output_path = os.getcwd() + '/output/task_3/'
cv2.imshow('img_l',key_point)
cv2.imwrite(output_path + 'feature_left.png',key_point)

feature_right= orb.detect(undistorted_right,None)
feature_right,feature_right1=orb.compute(undistorted_right,feature_right)
key_point_right = cv2.drawKeypoints(undistorted_right,feature_right,np.array([]),(0,250,0),0)


cv2.imshow('img_r',key_point_right)
cv2.imwrite(output_path + 'feature_right.png',key_point_right)


# Once a collection of features are obtained, match features on the two views using the OpenCV library "BFMatcher" class.
BFM = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# MATCHING AND DRAW MATCHES
matches = BFM.match(orb.compute(undistorted_left,feature_left)[1],orb.compute(undistorted_right,feature_right)[1])
matches = sorted(matches, key = lambda x:x.distance)
match_img = cv2.drawMatches(grady_scaled_left,feature_left,grady_scaled_right,feature_right,matches[:40],None,flags=2)
plt.imshow(match_img)
plt.axis('off')
plt.savefig(output_path + 'matched.png')
plt.show()

left = []
right= []
# for matching
for mat in matches:
    img2_match = mat.trainIdx
    img1_match = mat.queryIdx
    (x1, y1) = feature_left[img1_match].pt
    (x2, y2) = feature_right[img2_match].pt
    left.append((x1, y1))
    right.append((x2, y2))

points = cv2.triangulatePoints(P1,P2,np.array(left).T,np.array(right).T)

x = points[0]/points[3]
y = points[1]/points[3]
z = points[2]/points[3]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x, y, z,c='b', marker='o')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.savefig(output_path + '3D_points.png')
plt.show()