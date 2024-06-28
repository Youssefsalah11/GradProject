import socket
import struct

# Define host and port
HOST = '0.0.0.0'
PORT = 4889

#!/usr/bin/env python
import rospy
from sensor_msgs.msg import LaserScan
conn=None
# Parameters for obstacle detection
FRONT_ANGLE_START = -150  # Start angle for front obstacle detection (in degrees)
FRONT_ANGLE_END = 150   # End angle for front obstacle detection (in degrees)
OBSTACLE_DISTANCE_THRESHOLD = 0.5  # Distance threshold in meters (1 meter)

def detect_obstacle(ranges, angle_min, angle_increment):

    for i in range(len(ranges)):
        angle = angle_min + i * angle_increment
        angle_deg = angle * (180.0 / 3.14159)  # Convert radian to degree
        distance = ranges[i]
	
        if ((FRONT_ANGLE_START >= angle_deg) or  (angle_deg>= FRONT_ANGLE_END)):
            if distance > 0 and distance < OBSTACLE_DISTANCE_THRESHOLD:
                print(distance,angle_deg)
                return True
    return False

def scan_callback(msg):
    global conn
    # This function is triggered each time new scan data is received.
    if detect_obstacle(msg.ranges, msg.angle_min, msg.angle_increment):
        print("Obstacle detected! Stop the vehicle!")
        command = struct.pack('B', int(1))
        conn.send(command)
    else:
        print("Path is clear.")
        command = struct.pack('B', int(0))
        conn.send(command)

def listener():
    rospy.init_node('lidar_listener', anonymous=True)  # Initialize the ROS node
    rospy.Subscriber("/scan", LaserScan, scan_callback)  # Subscribe to the /scan topic
    print("Listening to /scan topic...")
    rospy.spin()  # Keeps the script from exiting until the node is stopped
# Create a socket object
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    # Bind the socket to the host and port
    server_socket.bind((HOST, PORT))
    a, b = server_socket.getsockname()
    print(f"{b}")

    # Listen for incoming connections
    server_socket.listen()
    print(f"Server listening on {HOST}:{PORT}")
    
    # Accept incoming connections
    conn, addr = server_socket.accept()
    with conn:
        print(f"Connected by {addr}")
        conn.send(bytes("hi", "utf-8"))
        while True:
            # Receive data from the client
            # data = conn.recv(4)
            # if not data:
            #     break
            
            # # Process received data
            # for byte in data:
            #     # Convert each byte to an integer
            #     integer_value = int(byte)
            #     print(f"Received: {integer_value}")
            
            # Send a resp
            listener()		
            user_input = input("Enter command to send to client (or 'exit' to quit): ")
            user_input1 = input("Enter command to send to client (or 'exit' to quit): ")

            if user_input.lower() == 'exit':
                break
            # Convert the user input to an integer and then to bytes
            try:
                command = struct.pack('B', int(1), int(user_input1))
                conn.send(command)
            except ValueError:
                print("Please enter a valid integer command.")
        conn.send(bytes("bye", "utf-8"))

            #conn.send(3)
            # conn.send(struct.pack('B', 120))
            # conn.send(bytes("bye from the server!", "utf-8"))
