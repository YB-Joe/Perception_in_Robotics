#!/usr/bin/env python

from __future__ import print_function
import roslib
roslib.load_manifest('pursuit_evasion')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import sys
import os
from darkflow.net.build import TFNet
import time
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, PoseStamped
from move_base_msgs.msg import MoveBaseActionResult, MoveBaseGoal
import actionlib
from actionlib_msgs.msg import *


tfnet = TFNet({"model": "/home/jj/catkin_ws/src/pursuit_evasion/darkflow/cfg/yolo.cfg", "load": "/home/jj/catkin_ws/src/pursuit_evasion/darkflow/cfg/yolo.weights", "threshold": 0.1,"config":"/home/jj/catkin_ws/src/pursuit_evasion/darkflow/cfg/"})

class image_converter: 
	def __init__(self):
		self.image_pub = rospy.Publisher("/tb3_0/tracking",Image)
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/tb3_0/camera/rbg/image_raw",Image,self.callback)
		self.bot_pose = rospy.Subscriber("/tb3_0/amcl_pose",PoseWithCovarianceStamped,self.initial_pose_2)
	def callback(self,data):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e) 
		img = np.asarray(cv_image)
		results = tfnet.return_predict(img)
    	new = np.copy(cv_image)
		results= results[0]
		top_x = int(results['topleft']['x'])
        top_y = int(results['topleft']['y'])

		w, h= int(results['bottomright']['x'])- top_x, int(results['bottomright']['y'])- top_y
		print("show me the results!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		print(results)
		if results['label']=='person':
			goal = PoseStamped()
			pub = rospy.Publisher('/tb3_0/move_base_simple/goal', PoseStamped, queue_size=100)

			if 300 <=  float((top_x + int(results['bottomright']['x']))/2) <= 400:
				goal.header.frame_id = "map"
				goal.pose.orientation.z = 0
				pub.publish(goal)
			elif  float((top_x + int(results['bottomright']['x']))/2) < 400:
				goal.header.frame_id = "map"
				goal.pose.orientation.z = 1
				pub.publish(goal)
			else:
				goal.header.frame_id = "map"
				goal.pose.orientation.z = -1
				pub.publish(goal)

        	confidence = results['confidence']
        	label = results['label'] + " " + str(round(confidence, 3))
		try:
			self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
		except CvBridgeError as e:
			print(e)
	def initial_pose_2(self, initial_pose):
        self.initial_pose = initial_pose

def main(args):
	image1 = image_converter()
    rospy.init_node('image_converter', anonymous=True)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down") 

if __name__ == '__main__':
	main(sys.argv)

