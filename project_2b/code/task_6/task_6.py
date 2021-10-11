import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from cv2 import aruco
import csv

if __name__ == "__main__":
    '''
    Step (1): Load the image sequence and camera parameters. 
    This is the same as the first step in the previous task. 
    In this task, one single image is processed at a time. 
    It is recommended to load all of them taken by one camera and process them one by one.
    
    '''
    left_intrinsics = np.array([[423.27381306, 0, 341.34626532],
                                [0, 421.27401756, 269.28542111],
                                [0, 0, 1]])

    dist1 = np.array([-4.33E-01, 2.64E-01, -3.41E-04, 5.06E-04, -1.08E-01])

    trans_vec = []
    rot = []
    rot_vec = []
    pos = []

    fig = plt.figure()
    plt.axis('equal')
    ax = fig.add_subplot(111, projection='3d')
    '''
    Step (2): Detect the ArUco marker.
    from cv2 import aruco
    '''
    for i in range(11):
        img = cv2.imread("../../images/task_6/left_" + str(i) + ".png")
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250) #dictionary size of 250
        corners, none, none = aruco.detectMarkers(img, aruco_dict)

        csv_file_name = '../../parameters/left_'+str(i)
        np.savetxt('{}.csv'.format(csv_file_name), np.asarray(corners[0][0]), delimiter=',', fmt='%f')

        img_marker = aruco.drawDetectedMarkers(img, corners)
        cv2.imshow("Detecting Marker on image " + str(i), img_marker)
        cv2.waitKey(0)
        cv2.imwrite("../../output/task_6/Marker on image left_" + str(i) + ".png",img_marker)

        objpoints = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]], dtype=np.float32)
        '''
        Step (3): Estimate the camera pose using PNP. Once the marker is detected, the 2D pixel coordinates of the four corners of the markers are known.
        '''
        retval, rtvec, tvec = cv2.solvePnP(objpoints, corners[0][0], left_intrinsics, dist1)

        #scaling factor = 4
        tvec = 4 * tvec

        trans_vec.append(tvec)
        rot.append(rtvec)
        RM, a = cv2.Rodrigues(rtvec)
        camera_position = -np.mat(RM).T * np.mat(tvec)

        rot_vec.append(RM)
        ax.scatter(camera_position[0], camera_position[1], camera_position[2], c='b', marker='^', s = 0, zdir = 'z')
        ax.text(camera_position[0][0,0], camera_position[1][0,0] + 0.01, camera_position[2][0,0], i , zdir='y')
        pos.append(camera_position)

        print("\n Image left_"+ str(i))
        print("R : ", RM.T)
        print("\n T : ", camera_position)
        print("\n")
        # The pose of the camera in 11 views using "left_0.png" to "left_10.png".
        # Two different perspectives are illustrated here to show the 3D relation.
        pyramid = np.array([[-1, 1, 1, -1, 0], [-1, -1, 1, 1, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
        points = np.hstack((RM.T, camera_position))
        points = np.append(points, [[0, 0, 0, 1]], axis=0)
        #  Note that you do not need to draw the marker in 3D. You can just draw a square with a red circle on the top-left corner to represent the marker
        pyramid_matmul = np.delete(np.mat(points) * np.mat(pyramid), 3, 0)

        result = [[np.array(pyramid_matmul.T)[0], np.array(pyramid_matmul.T)[1], np.array(pyramid_matmul.T)[4]],
                    [np.array(pyramid_matmul.T)[0], np.array(pyramid_matmul.T)[3], np.array(pyramid_matmul.T)[4]],
                    [np.array(pyramid_matmul.T)[2], np.array(pyramid_matmul.T)[1], np.array(pyramid_matmul.T)[4]],
                    [np.array(pyramid_matmul.T)[2], np.array(pyramid_matmul.T)[3], np.array(pyramid_matmul.T)[4]],
                    [np.array(pyramid_matmul.T)[0], np.array(pyramid_matmul.T)[1], np.array(pyramid_matmul.T)[2],
                     np.array(pyramid_matmul.T)[3]]]

        ax.add_collection3d(Poly3DCollection(result, facecolors='white', linewidths=1, edgecolors='blue', alpha=.25))

    ax.add_collection3d(Poly3DCollection([[np.array([0,5,0]), np.array([5,5,0]), np.array([5,0,0]), np.array([0,0,0])]],
                                             facecolors='black', linewidths=1, edgecolors='k', alpha=.5))

    ax.add_collection3d(Poly3DCollection([[np.array([0,1,0]), np.array([1,1,0]), np.array([1,0,0]), np.array([0,0,0])]],
                                             facecolors='red', linewidths=1, edgecolors='r', alpha=1))
    ax.set_xlim(-15, 15)
    ax.set_ylim(10,-10)
    ax.set_zlim(20,-20)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()

