#!/usr/bin/env python 
import rospy
from geometry_msgs.msg import Twist

PI=3.1415926535897
is_Forward = True
clockwise = 0

a_speed = 60*2*PI/360 #angular speed and convert degree to radian

#Starts a new node
rospy.init_node('robot_cleaner', anonymous=True)
velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
velocity_msg = Twist()

velocity_msg.linear.y = 0
velocity_msg.linear.z = 0
velocity_msg.angular.x = 0
velocity_msg.angular.y = 0
velocity_msg.angular.z = -1.0
    
def move(distance): #Move the turtlebot
		
	#Since we are moving just in x-axis
	velocity_msg.linear.x = 2.0

	#Setting the current time for distance calculus
	t0 = rospy.Time.now().to_sec()
	current_distance = 0 
  		   
	#Loop to move the turtle in an specified distance
	while(current_distance < distance):
		
		#Publish the velocity
		velocity_publisher.publish(velocity_msg)
		#Takes actual time to velocity calculus
		t1=rospy.Time.now().to_sec()
		#Calculates distancePoseStamped
		current_distance= 1*(t1-t0)
		
	#FOrcing turtlebot to stop
	velocity_msg.linear.x = 0.0

def rotate(angle):

	#We wont use linear components
	velocity_msg.angular.z = -1.2

	# Setting the current time for distance calculus
	t0 = rospy.Time.now().to_sec()
	current_angle = 0

	while(current_angle < angle):
		#publish the angle
		velocity_publisher.publish(velocity_msg)
		#Takes actual time to angular calculus
		t1 = rospy.Time.now().to_sec()
		#Calculates anglePoseStamped
		current_angle = a_speed*(t1-t0)

	#Forcing our robot to stop
	velocity_msg.angular.z = -1

   
if __name__ == '__main__':
	try:
		#Implementing the functions
		#First letter of my name : C
		rotate(120*2*PI/360)  
		move(2)
		rotate(-90*2*PI/360) 
		move(2)
		

	except rospy.ROSInterruptException: pass
