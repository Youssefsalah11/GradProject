#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped

def pose_callback(data):
    # Extract the position information
    position = data.pose.position
    orientation = data.pose.orientation

    # Print the position
    rospy.loginfo("Current position: x=%f, y=%f, z=%f", position.x, position.y, position.z)
    rospy.loginfo("Current orientation: x=%f, y=%f, z=%f, w=%f", orientation.x, orientation.y, orientation.z, orientation.w)

def listener():
    rospy.init_node('pose_listener', anonymous=True)
    rospy.Subscriber("/slam_out_pose", PoseStamped, pose_callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
