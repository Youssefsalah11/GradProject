import cv2
import time
import torch
import numpy as np
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
import socket
import signal
import sys

# Define host and port
HOST = '0.0.0.0'
PORT = 4888


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    # Place any code here that you want to execute on Ctrl+C
    conn.close()
    sys.exit(0) 

# Attach the signal handler
signal.signal(signal.SIGINT, signal_handler)

def send_command(num):
    command = int(num)
    conn.send(command.to_bytes(1, 'big'))
    
    

# Load the model
weights = 'signweights.pt'  # path to custom weights file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(weights, device=device, dnn=False)
stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
imgsz = 448
model.warmup()  # warmu
#print(names)
# Access the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Couldn't open the webcam")
    exit()

# Variables for calculating FPS
start_time = time.time()
frame_count = 0

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


        # Loop to continuously capture frames from the webcam
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Perform object detection on the frame
            original_frame = frame.copy()
            orig_h, orig_w = original_frame.shape[:2]

            # Resize the frame to the input size of the model
            frame_resized = cv2.resize(original_frame, (imgsz, imgsz))

            # Convert BGR to RGB
            img = frame_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x448x448
            img = np.ascontiguousarray(img)

            # Inference
            img = torch.from_numpy(img).to(device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment=False, visualize=False)

            # Apply NMS (Non-Maximum Suppression)
            pred = non_max_suppression(pred, 0.7, 0.45, None, False, max_det=1000)

            # Process detections and rescale to original dimensions
            for i, det in enumerate(pred):  # per image
                if len(det):
                    # Rescale boxes from imgsz to original frame size
                    det[:, [0, 2]] *= orig_w / imgsz  # Scale x coordinates
                    det[:, [1, 3]] *= orig_h / imgsz  # Scale y coordinates

        #           annotator = Annotator(original_frame, line_width=2, example=str(names))
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = f'{names[c]} {conf:.2f}'
                        print(label)
                        try:
                            if names[c] == 'stop' or names[c] == 'trafic light- red' or names[c] =='traffic light- red':
                                send_command('0')
                            elif names[c] == 'traffic light- green':
                                send_command('2')
                            elif names[c] == 'speed limit -30-':
                                send_command('30')
                            elif names[c] == 'speed limit -60-':
                                send_command('60')
                            elif names[c] == 'speed limit -90-':
                                send_command('90')
                            elif names[c] == 'crosswalk': 
                                send_command('30')
                            else:
                                pass
                        except:
                            print("Please enter a valid integer command.")
        #              annotator.box_label(xyxy, label, color=colors(c, True))

            # Display the captured frame with detections
        # cv2.imshow('Webcam with Detections', original_frame)
            # Increment frame count
            frame_count += 1

            # Calculate FPS every second
            #if time.time() - start_time >= 1:
                #fps = frame_count / (time.time() - start_time)
                #print(f"FPS: {fps:.2f}")
                #start_time = time.time()
                #frame_count = 0

            # Break the loop when 'q' is pressed
            cv2.waitKey(1000)

        # Release the webcam and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

