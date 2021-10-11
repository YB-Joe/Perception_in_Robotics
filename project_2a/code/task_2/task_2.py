import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

base_folder = os.getcwd() +'/parameters/'

s = cv2.FileStorage(base_folder + 'left_camera_intrinsics.xml', cv2.FileStorage_READ)

mtxl = s.getNode('mtx_left').mat()
distl = s.getNode('distCoeffs_left').mat()
s.release()

s = cv2.FileStorage(base_folder + 'right_camera_intrinsics.xml', cv2.FileStorage_READ)
mtxr = s.getNode('mtx_right').mat()
distr = s.getNode('distCoeffs_right').mat()
s.release()


# row anc column
row = 9
col = 6

objp = np.zeros((row*col,3), np.float32)
objp[:,:2] = np.mgrid[0:col,0:row].T.reshape(-1,2)

# store the obj points

# 3d point
obj = []
imgpoints_left = []
# 2d points in image plane.
imgpoints_right = []

img_folder = os.getcwd() + '/'

img1 = cv2.imread(img_folder + 'images/task_2/left_0.png')
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

# Find the chess board corners
# similar as task1
ret1, corners1 = cv2.findChessboardCorners(gray1, (col,row),None)
obj.append(objp)

# If found, add object points, image points (after refining them)
if ret1 == True:
    imgpoints_left.append(corners1)
    corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1),(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    img1 = cv2.drawChessboardCorners(img1, (col,row), corners1,ret1)
    cv2.drawChessboardCorners(img1, (row, col), corners1, ret1)
    cv2.imshow('img', img1)
    cv2.imwrite(img_folder + '/output/task_2/original2.png', img1)


img2= cv2.imread(img_folder + 'images/task_2/right_0.png')
gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# Find the chess board corners
ret2, corners2 = cv2.findChessboardCorners(gray2, (col,row),None)

# If found, add object points, image points (after refining them)
if ret2 == True:
    imgpoints_right.append(corners2)
    corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    cv2.drawChessboardCorners(img2, (row, col), corners2, ret2)
    cv2.imshow('img', img2)
    cv2.imwrite(img_folder + '/output/task_2/original.png', img2)
height,width = img2.shape[:2]

T = np.zeros((3, 1), dtype=np.float64)
R = np.identity(3, dtype = np.float64)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 1e-5)

retval, mtxl, distl, mtxr, distr, R, T, E, F=cv2.stereoCalibrate\
    (obj,imgpoints_left,imgpoints_right,mtxl,distl,mtxr,distr,\
        (width,height),R,T,criteria = criteria,flags=cv2.CALIB_FIX_INTRINSIC)

path1 = os.getcwd() + '/'
s = cv2.FileStorage(path1 + 'parameters/stereo_calibration.xml', cv2.FileStorage_WRITE)
# calibration
#s.write('Camera_Matrix_1', mtxl)
#s.write('Distortion_Coefficient_1', distl)
#s.write('Camera_Matrix_2', mtxr)
#s.write('Distortion_Coefficient_2', distr)
s.write('R', R)
s.write('T', T)
s.write('E', E)
s.write('F', F)
s.release()


img_left=np.array(imgpoints_left)
img_right=np.array(imgpoints_right)
img_left = img_left.reshape(54,1,2)
img_right = img_right.reshape(54,1,2)

img_left = cv2.undistortPoints(img_left,mtxl,distl)
img_right = cv2.undistortPoints(img_right,mtxr,distr)


I = np.identity(3, dtype=np.float64)
O = np.zeros((3, 1), dtype=np.float64)

P1=np.hstack((I,O))
P2=np.hstack((R,T))

s = cv2.FileStorage(path1 + 'parameters/P1.xml', cv2.FileStorage_WRITE)
s.write('P1', P1)
s.release()
s = cv2.FileStorage(path1 + 'parameters/P2.xml', cv2.FileStorage_WRITE)
s.write('P2', P2)
s.release()

points=cv2.triangulatePoints(P1,P2,img_left,img_right)


fig = plt.figure()
ax = plt.axes(projection='3d')
plt.title('Before Rectification')
plt.savefig(img_folder + '/output/task_2/before.png')

# inverse pyramid
prep = np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1],  [-1, 1, 1], [1, 1, 1]])
ax.scatter3D(prep[:, 0], prep[:, 1], prep[:, 2])

pyramid = [[[-1, -1, 1],[1, -1, 1],[0, 0, 0]], [[-1, -1, 1],[-1, 1, 1],[0,0,0]],[[1, 1, 1],[1, -1, 1],[0,0,0]], [[1,1,1],[-1, 1, 1],[0,0,0]], [[-1, -1, 1],[1, -1, 1],[1, 1, 1],[-1, 1, 1]]]
#plot
ax.add_collection3d(Poly3DCollection(pyramid, linewidths=1, edgecolors='k'))

prep2 = np.array([[1.4264, -0.9801, 0.9212], [3.4254, -0.9930, 0.9828], [3.4380, 1.0069, 0.9946],  [1.4390, 1.0198, 0.9330], [2.4630, 0.0191, -0.0416]])
ax.scatter3D(prep2[:, 0], prep2[:, 1], prep2[:, 2])

pyramid2 = [ [[1.4264, -0.9801, 0.9212],[3.4254, -0.9930, 0.9828],[2.4630, 0.0191, -0.0416]], [[1.4264, -0.9801, 0.9212],[1.4390, 1.0198, 0.9330],[2.4630, 0.0191, -0.0416]],
 [[3.4380, 1.0069, 0.9946],[3.4254, -0.9930, 0.9828],[2.4630, 0.0191, -0.0416]], [[3.4380, 1.0069, 0.9946],[1.4390, 1.0198, 0.9330],[2.4630, 0.0191, -0.0416]], [[1.4264, -0.9801, 0.9212],[3.4254, -0.9930, 0.9828],[3.4380, 1.0069, 0.9946],[1.4390, 1.0198, 0.9330]]]

ax.add_collection3d(Poly3DCollection(pyramid2, linewidths=1, edgecolors='k'))

# x,y,z coordinates
x=points[0]/points[3]
y=points[1]/points[3]
z=points[2]/points[3]
ax.scatter3D(x, y, z,c=z, marker='o')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
#range
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
ax.set_zlim(-10,10)
plt.show()


fig = plt.figure()
ax = plt.axes(projection='3d')
plt.savefig(img_folder + '/output/task_2/AFTER.png')
plt.title('AFTER Rectified')

# pyramid
prep3 = np.array([[-1.0245, -0.9893, 0.9858], [0.9752, -1.0047, 1.0196], [0.9907, 0.9952, 1.0140],  [-1.0090, 1.0106, 0.9801], [0, 0, 0]])
ax.scatter3D(prep3[:, 0], prep3[:, 1], prep3[:, 2])

pyramid3 = [[prep3[0],prep3[1],prep3[4]], [prep3[0],prep3[3],prep3[4]], [prep3[2],prep3[1],prep3[4]], [prep3[2],prep3[3],prep3[4]], [prep3[0],prep3[1],prep3[2],prep3[3]]]

# plotting
ax.add_collection3d(Poly3DCollection(pyramid3,facecolors='white', linewidths=1, edgecolors='k'))


prep4 = np.array([[1.3670, -1.0024 , 0.9850], [3.3604, -1.0436, 1.1418], [3.4003, 0.9559, 1.1607],  [1.4069, 0.9971, 1.0039], [2.4622, -0.0154, 0.0760]])
ax.scatter3D(prep4[:, 0], prep4[:, 1], prep4[:, 2])

pyramid4 = [[prep4[0],prep4[1],prep4[4]], [prep4[0],prep4[3],prep4[4]],[prep4[2],prep4[1],prep4[4]], [prep4[2],prep4[3],prep4[4]], [prep4[0],prep4[1],prep4[2],prep4[3]]]

ax.add_collection3d(Poly3DCollection(pyramid4,facecolors='white', linewidths=1, edgecolors='k'))

x=points[0]/points[3]
y=points[1]/points[3]
z=points[2]/points[3]
ax.scatter3D(x, y, z,c=z, marker='o')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
ax.set_zlim(-10,10)

plt.show()

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtxl,distl,mtxr,distr,img2.shape[:2],R,T)

s = cv2.FileStorage(path1 + 'parameters/stereo_rectification.xml', cv2.FileStorage_WRITE)
# rectification
s.write('R1', R1)
s.write('R2', R2)
s.write('Q', Q)
s.write('roi1', roi1)
s.write('roi2', roi2)

s.release()

hegiht = 480
width = 680

mapx1, mapy1 = cv2.initUndistortRectifyMap(mtxl,distl,R1,mtxl,(width,height),cv2.CV_32FC1)
mapx2, mapy2 = cv2.initUndistortRectifyMap(mtxr,distr,R2,mtxr,(width,height),cv2.CV_32FC1)
mapx11, mapy11 = cv2.initUndistortRectifyMap(mtxl,distl,None,mtxl,(width,height),cv2.CV_32FC1)
mapx22, mapy22= cv2.initUndistortRectifyMap(mtxr,distr,None,mtxr,(width,height),cv2.CV_32FC1)


img_rect1 = cv2.remap(img1, mapx1, mapy1, cv2.INTER_LINEAR)
img_rect2 = cv2.remap(img2, mapx2, mapy2, cv2.INTER_LINEAR)
img_undis_unrect1 = cv2.remap(img1, mapx11, mapy11, cv2.INTER_LINEAR)
img_undis_unrect2 = cv2.remap(img2, mapx22, mapy22, cv2.INTER_LINEAR)
# cropping
w = max(img_rect1.shape[0], img_rect2.shape[0])
h = img_rect1.shape[1] + img_rect2.shape[1]
output_path = os.getcwd() + '/output/task_2/'
cv2.imwrite(output_path + 'undistorted_unrectification_1.png',img_undis_unrect1)
cv2.imwrite(output_path + 'undistorted_unrectification_2.png', img_undis_unrect2)

size = (w, h, 3)
img = np.zeros(size, dtype=np.uint8)
img[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
img[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2

cv2.imshow('img_Rectified', img)
cv2.imwrite(output_path + '/image_rectified.png', img)

