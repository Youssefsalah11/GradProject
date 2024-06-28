from PIL import Image
import numpy as np
import cv2

import math as mt
import socket
import signal
import struct
import sys
import time
import requests


HOST = '0.0.0.0'
PORT = 4891

curve = 0
sleep = True

def send_command(sign, angle,conn):
    angle = angle*-1 if sign else angle
    command = struct.pack('2B', int(sign), int(angle))
    conn.send(command)

    
def call_server_function(frame):
    url = 'http://192.168.1.3:5003/call_function'
       
    # Encode the frame as JPEG
    _, img_encoded = cv2.imencode('.jpg', frame)
    
    # Convert the encoded image to bytes
    files = {'image': img_encoded.tobytes()}
    
    # Send the POST request
    response = requests.post(url, files=files)

    
    if response.status_code == 200:
        result = response.json()
        return result['result']
    else:
        return f"Error: {response.status_code}"





cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Couldn't open the webcam")
    exit()


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

        ##change here and in controler
        max_velocity = 3
        average_wait = 2
        count = 0
        steering_angle = 0
        ## fix error 
        while True:
            try:


                ret, frame = cap.read()
                curve2 =call_server_function(frame)
                print(curve2)

                # # curve2 = sum(saved_readings)/len(saved_readings)
                # if curve2 < -20:
                #     curve2 = -20
                # elif curve2 > 20:
                #     curve2 = 20
                # steering_angle = curve2 / 20* 70                    
                # print(f"Control Actions: {uv/max_velocity*100, udelta*360/(2*mt.pi)}\n")
                # print(f"Control Actions: { cur}\n")

                # send_command(uv/max_velocity*100,udelta*360/(2*mt.pi),conn)
                sign = 0 if curve2 > 0 else 1
                send_command(sign,curve2,conn)


            except Exception as e:
                print("error in controler",e)
# if __name__ == '__main__':
#     control_thread = threading.Thread(target = control, args= (2, 5))
#     control_thread.start()
