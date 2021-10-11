import argparse
import matplotlib.pyplot as plt
import os
from mpl_toolkits import mplot3d
import numpy as np
import cv2
import glob
import csv
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import math

if __name__ == "__main__":


    left_intrinsics = np.array([[423.27381306, 0, 341.34626532],
                         [0, 421.27401756, 269.28542111],
                         [0, 0, 1]])

    dist1 = np.array([-4.33E-01, 2.64E-01, -3.41E-04, 5.06E-04, -1.08E-01 ])

    right_intrinsics = np.array([[420.91160482, 0, 352.16135589],
                      [0, 418.72245958, 264.50726699],
                      [0, 0, 1]])
    dist2 = np.array([-0.412, 0.198, -4.73E-04, -0.00078, -5.08E-02])#np.reshape([parameter1[5]],(1,5)).astype('float64')


    #left and right images
    left_image1 = cv2.imread("../../images/task_7/left_1.png")
    left_image1 = cv2.cvtColor(left_image1, cv2.COLOR_BGR2GRAY)

    left_image2 = cv2.imread("../../images/task_7/left_3.png")
    left_image2 = cv2.cvtColor(left_image2, cv2.COLOR_BGR2GRAY)

    # undistort image
    dst = cv2.undistort(left_image1, left_intrinsics, dist1, None, left_intrinsics)

    # undistort image
    dst2 = cv2.undistort(left_image2, left_intrinsics, dist1, None, left_intrinsics)

    orb = cv2.ORB_create()
    kps = orb.detect(dst, None)
    kps1, des1 = orb.compute(dst, kps)

    kps_image2 = orb.detect(dst2, None)
    kps2, des2 = orb.compute(dst2, kps_image2)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    # Sort in order of distances.
    matches = sorted(matches, key=lambda x: x.distance)

    img3 = cv2.drawMatches(dst, kps1, dst2, kps2, matches, None, flags=2)

    cv2.imshow('img3',img3)
    cv2.imwrite("../../output/task_7/Match.jpg",img3)

    left_image_points = []
    right_image_points = []

    for match in matches:
        left_image_points.append(kps1[match.queryIdx].pt)
        right_image_points.append(kps2[match.queryIdx].pt)

    kp1_match = []
    kp2_match = []

    for m in matches:
        kp1_match.append(kps1[m.queryIdx].pt)
        kp2_match.append(kps2[m.trainIdx].pt)

    kp1_undistort = cv2.undistortPoints(np.expand_dims(kp1_match, axis=1), left_intrinsics, dist1)
    kp2_undistort = cv2.undistortPoints(np.expand_dims(kp2_match, axis=1), left_intrinsics, dist1)

    E, mask0 = cv2.findEssentialMat(kp1_undistort, kp2_undistort, method=cv2.RANSAC, prob=0.99, threshold=0.001)

    mask_bool=mask0.astype(bool)

    img6 = cv2.drawMatches(dst, kps1, dst2, kps2, matches, None,flags=2)

    cv2.imshow("inliners", img6)
    cv2.imwrite("../../output/task_7/inliers.jpg",img6)
    #Recover the pose
    points, R, t, mask1 = cv2.recoverPose(E, kp1_undistort, kp2_undistort)

    #Tringulate the points
    proj1 = np.dot(left_intrinsics,  np.hstack((np.eye(3, 3), np.zeros((3, 1)))))
    proj2 = np.dot(right_intrinsics,  np.hstack((R, t)))
    triangulate = cv2.triangulatePoints(proj1, proj2, np.expand_dims(left_image_points, axis=1), np.expand_dims(right_image_points, axis=1))
    point_3d = (triangulate / np.tile(triangulate[-1, :], (4, 1))).T


    v = np.float64([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

    cnr1= np.loadtxt('../../parameters/left_1.csv',delimiter=',')
    cnr2= np.loadtxt('../../parameters/left_3.csv',delimiter=',')

    ret1, rvec1, tvec1 = cv2.solvePnP(v, cnr1, left_intrinsics, dist1)
    ret2, rvec2, tvec2 = cv2.solvePnP(v, cnr2, left_intrinsics, dist1)

    tvec1 = 4 * tvec1  # Scaling
    tvec2 = 4 * tvec2  # Scaling

    r1, none = cv2.Rodrigues(rvec1)
    r2, none = cv2.Rodrigues(rvec2)

    camera_position1 = -np.mat(r1).T * np.mat(tvec1)
    camera_position2 = -np.mat(r2).T * np.mat(tvec2)

    # plotting the 3D points with pyramid

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(point_3d[:,2], point_3d[:,0], point_3d[:,1], c='b', marker='.')

    v = np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1], [0, 0, 0]])
    v = v.T
    v = np.append(v, [[1, 1, 1, 1, 1]], axis=0)
    H = np.hstack((r1.T, camera_position1))
    H = np.append(H, [[0, 0, 0, 1]], axis=0)
    v_t = (np.mat(H) * np.mat(v))
    v_t = np.delete(v_t, 3, 0)

    v = np.array(v_t.T) * 3

    verts = [[v[0], v[1], v[4]], [v[0], v[3], v[4]],
             [v[2], v[1], v[4]], [v[2], v[3], v[4]], [v[0], v[1], v[2], v[3]]]

    ax.add_collection3d(Poly3DCollection(verts, facecolors='white', linewidths=1, edgecolors='blue', alpha=.25))

    v2 = np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1], [0, 0, 0]])
    v2 = v2.T
    v2 = np.append(v2, [[1, 1, 1, 1, 1]], axis=0)
    H1 = np.hstack((r2.T, camera_position2))
    H1 = np.append(H1, [[0, 0, 0, 1]], axis=0)
    v_t1 = (np.mat(H1) * np.mat(v2))
    v_t1 = np.delete(v_t1, 3, 0)
    v2 = np.array(v_t.T) * 3
    verts1 = [[v2[0], v2[1], v2[4]], [v2[0], v2[3], v2[4]],
             [v2[2], v2[1], v2[4]], [v2[2], v2[3], v2[4]], [v2[0], v2[1], v2[2], v2[3]]]
    ax.add_collection3d(Poly3DCollection(verts1, facecolors='white', linewidths=1, edgecolors='blue', alpha=.25))

    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_zlabel('X')
    plt.title('3D point')
    plt.show()



