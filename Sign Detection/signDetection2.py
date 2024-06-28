import cv2
import time
import torch
import numpy as np
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
#
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
                print(names)
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

