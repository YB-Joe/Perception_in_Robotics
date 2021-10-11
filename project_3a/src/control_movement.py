#! /usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import math
import time


x = 0
y = 0
z = 0
yaw = 0

def pose(pose_message):
    global x, y, z, yaw
    x = pose_message.pose.pose.position.x
    y = pose_message.pose.pose.position.y
    first = 2 * (pose_message.pose.pose.orientation.w * pose_message.pose.pose.orientation.z + pose_message.pose.pose.orientation.x * 				pose_message.pose.pose.orientation.y)
    second = 1 - 2 * (pose_message.pose.pose.orientation.y * pose_message.pose.pose.orientation.y + pose_message.pose.pose.orientation.z * 				pose_message.pose.pose.orientation.z)
    yaw = math.atan2(first, second)

def move(speed, distance, forward):
    global x, y
    msg = Twist()
    glb_x = x
    glb_y = y
    if forward:
        msg.linear.x = abs(speed)
    else:
        msg.linear.x = -abs(speed)
    distance_moved = 0
    loop = rospy.Rate(10)
    topic = '/cmd_vel'
    velocity_publisher = rospy.Publisher(topic, Twist, queue_size = 10)

    while True:
        rospy.loginfo('Turtlebot is moving')
        velocity_publisher.publish(msg)
        rate.sleep()
        distance_moved += abs(math.sqrt(((x - glb_x)**2) + ((y - glb_y)**2)))
        if not (distance_moved < distance):
            rospy.loginfo("Turtlebot-arrival at destination")
            break
   
    msg.linear.x = 0
    velocity_publisher.publish(msg)

def rotate(speed, angle, clockwise):
    msg = Twist()
    global yaw

    glb_yaw = yaw

    if clockwise:
        msg.angular.z = -math.radians(abs(speed))
    else:
        msg.angular.z = math.radians(abs(speed))

    loop_rate = rospy.Rate(10)
    topic = '/cmd_vel'
    velocity_publisher = rospy.Publisher(topic, Twist, queue_size = 10)

    timeone = rospy.Time.now().to_sec()

    while True:
        rospy.loginfo('Turtlebot is rotating')
        velocity_publisher.publish(msg)

        loop_rate.sleep()

        current_angle = abs((rospy.Time.now().to_sec() - timeone) * math.radians(speed))

        if not (current_angle < math.radians(angle)):
            rospy.loginfo("turtlebot has rotated")
            break
    msg.angular.z = 0
    velocity_publisher.publish(msg)
    

def supple(speed, speed, angle, clockwise):
    msg = Twist()
    global x, y, yaw

    glb_yaw = yaw

    if clockwise:
        msg.linear.x = abs(speed)
        msg.angular.z = -abs(math.radians(abs(speed)))
    else:
        msg.linear.x = abs(speed)
        msg.angular.z = abs(math.radians(abs(speed)))

    
    
    loop_rate = rospy.Rate(10)
    topic = '/cmd_vel'
    velocity_publisher = rospy.Publisher(topic, Twist, queue_size = 10)

    tn = rospy.Time.now().to_sec()

    while True:
        velocity_publisher.publish(msg)
        
        loop_rate.sleep()

        current_angle = abs((rospy.Time.now().to_sec() - tn) * math.radians(speed))

        if not (current_angle < math.radians(angle)):
            break
    msg.linear.x = 0
    msg.angular.z = -1
    velocity_publisher.publish(msg)
    

def travel_the_house():

    print("------------------Starting to travel the house------------------")

    rotation_speed = 20
    movement_speed = 0.4
    
    rotate(rotation_speed, 90, False)
    move(movement_speed, 350, True)

    rotate(rotation_speed, 90, False)
    move(movement_speed, 80, True)
    
    rotate(rotation_speed, 95, True)
    move(movement_speed, 1100, True)

    rotate(rotation_speed, 35, True)
    move(movement_speed, 150, True)

    rotate(rotation_speed, 30, True)
    move(movement_speed, 300, True)
    
    
    rotate(rotation_speed, 60, True)
    move(movement_speed, 350, True)
    
    
    print("------------------Finished travelling the house------------------")
    pass


def draw_initials():

    
    print ("Drawing of initials = C")
   
    # Drawing C(my initial)
    rotate(rotation_speed, 90, True)
    supple(0.6 * 3, 10 * 3, 120, False)
   
    move(0.6, 40, False)
    
    print ("Finished the Drawing of initials - ( C )")

if __name__ == '__main__':
    try:
        rospy.init_node('house_movement', anonymous = True)
        topic = '/cmd_vel'
        velocity_publisher = rospy.Publisher(topic, Twist, queue_size = 10)
        position = '/odom'
        pose_subscriber = rospy.Subscriber(position, Odometry, pose)

        print("------------------STart!!------------------")
      	travel_the_house()
	time.sleep(2)
	draw_initials()

    except rospy.ROSInterruptException:
        print("------------------DOne------------------")
        rospy.loginfo("Terminated")
