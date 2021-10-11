#! /usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()

def cvt2img(ros_image):
    global bridge
    try:
        cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    except CvBridgeError as e:
        print e
    resized = cv2.resize(cv_image, (0,0), fx = 0.60, fy = 0.60)
    cv2.imshow("Turtlebot's View", resized)
    cv2.waitKey(1)
if __name__ == '__main__':
    try:
        rospy.init_node('node_for_camera', anonymous = True)
        root = "/camera/rgb/image_raw"
        image_sub = rospy.Subscriber(root, Image, cvt2img)
        
        print("------------------Camera sensor started------------------")
        rospy.spin()

        if rospy.is_shutdown:
            print("------------------Camera sensor shutdown------------------")
            cv2.destroyAllWindows()
    except rospy.ROSInterruptException:
        rospy.loginfo("Finished!")
