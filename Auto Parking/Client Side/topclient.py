#!/usr/bin/env python3
import numpy as np
import math as mt
from box_finder import box_finder
from desired_motion import desired_motion, initialize_poses
import threading
from slots_data_finder import slots_data_finder
from pose_controler import pose_controller
from pose_errors import pose_errors
from update_vehicle import update_vehicle_pose
import matplotlib.pyplot as plt
import requests
import torchvision.transforms as T
from PIL import Image
import rospy
from geometry_msgs.msg import PoseStamped
import time
import socket
import signal
import struct

HOST = '0.0.0.0'
PORT = 4890

def send_command(speed, angle,sign,direction,conn):
    angle = angle*-1*360/3.14 if sign else angle*360/3.14
    speed = speed * -1 if direction else speed
    print(angle)
    command = struct.pack('4B', int(speed), int(angle),int(sign),int(direction))
    #print(command)
    conn.send(command)
    time.sleep(0.5)

    
# Global variables to store the pose
current_position = None
unsleep_flag = True

def pose_callback(data):
    # Extract the position information
    global current_position
    global unsleep_flag

    position = data.pose.position
    orientation = data.pose.orientation

    current_position = [position.x/0.05+64,position.y/0.05+64,orientation.z]
    unsleep_flag = False
    # Print the position
    # rospy.loginfo("Current position: x=%f, y=%f, z(orientation)=%f", position.x, position.y, orientation.z)
    #rospy.loginfo("Current orientation: x=%f, y=%f, z=%f, w=%f", orientation.x, orientation.y, orientation.z, orientation.w)

def listener():
    rospy.init_node('pose_listener', anonymous=True)
    rospy.Subscriber("/slam_out_pose", PoseStamped, pose_callback)
    #rospy.spin()

def check_pose_reached(errors, velocity, epsilon_p_max, v_p_max):
    return abs(errors[0]) <= epsilon_p_max and abs(errors[1]) <= v_p_max



def load_image(image_path):
    transforms = T.Compose([
        T.ToTensor(),
    ])
    image = Image.open(image_path).convert("L")
    image = transforms(image)
    return image


def main_simulation_loop(initial_position, dt, epsilon_p_max, v_p_max):
    global current_position
    global unsleep_flag

    current_position = list(initial_position)
    

    # Just an example of selecting pose ids in a sequential manner
    for pose_id in range(len(poses_data)):
        count = 0
        uv =0
        desired_pose = desired_motion(pose_id)

        while True:
            errors, direction = pose_errors(desired_pose, current_position)
            if check_pose_reached(errors, uv, epsilon_p_max, v_p_max):
                break
            
            uv, udelta = pose_controller(errors, float(direction))
            
            # Print the current state for monitoring
            # print(f"Step {count}:")
            print(f"Current Pose: {current_position}")
            # print(f"Desired Pose: {desired_pose}")
            print(f"Control Actions: {uv, udelta}\n")
            time.sleep(1)
            unsleep_flag = True




def call_server_function(image_path):
    url = 'http://192.168.1.116:5005/call_function'
       
        # Load the image file
    with open(image_path, 'rb') as img_file:
        files = {'image': img_file}
        response = requests.post(url, files=files)

    
    if response.status_code == 200:
        result = response.json()
        return result['result']
    else:
        return f"Error: {response.status_code}"

# listener()
def control(epsilon_p_max, v_p_max):
    # listener()
    global current_position
    global unsleep_flag
    # image = load_image("/home/adas/adas/occupancy_grid_raw_image.png")
    image_path ="/home/adas/adas/occupancy_grid_raw_image.png"
    # poses_data = call_server_function(image_path) 
    # poses_data = [{'end': [150, 20, 0.0107963267948966], 'mid': [256, 128, 0], 'start': [151.6769256591797, 89.84503173828125, 1.5707963267948966]}] 
    # poses_data=[{'end':[35,56,0]}]
    poses_data=[{'mid': [80, 62, -0.37], 'end1': [96, 55, -0.2],'end2':[89, 55 , -0.04]}]
    # print(poses_data)
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
            # conn.send(bytes("hi", "utf-8"))

            ##change here and in controler
            max_velocity = 2
            average_wait = 2
            count = 0
            steering_angle = 0

            if len(poses_data) == 0:
                raise ("no poses found")

            initialize_poses(poses_data)
            
            # initial_position = poses_data[0]['start']

            # poses_data = [poses_data[0]['start'],poses_data[0]['mid'],poses_data[0]['end']]

            #current_position = list(initial_position)
            initial_position = [128,128,0]
            

            # Just an example of selecting pose ids in a sequential manner
            for pose_id in range(3):
                count = 0
                uv = 0
                desired_pose = desired_motion(pose_id)

                while True:
                    try:
                        errors, direction = pose_errors(desired_pose, current_position)
                        if abs(current_position[0]-desired_pose[0])<= 2 and abs(current_position[1]-desired_pose[1])<= 5 and abs(current_position[2] - desired_pose[2]) <= 0.1:
                            break
                        
                        uv, udelta = pose_controller(errors, float(direction))
                        

                        print(f"Current Pose: {current_position}")
                        print(f"Desired Pose: {desired_pose}")
                        print(f"Control Actions: {uv, udelta}\n")
                        # print(f"erros:{errors}")
                        if abs(uv) > 16:
                            continue
                        send_command(uv, udelta, 1 if udelta<0 else 0, 1 if uv<0 else 0 ,conn)
                        unsleep_flag = True
                        time.sleep(0.2)
                        

                    except Exception as e:
                        print("error in controler",e)
                        print(traceback.format_exc())
            send_command(0, 0, 1 if udelta<0 else 0, 1 if uv<0 else 0 ,conn)
#print("hi")s

#control()
if __name__ == '__main__':
    control_thread = threading.Thread(target = control, args = (4,1))
    control_thread.start()
    listener()
 #    control()

