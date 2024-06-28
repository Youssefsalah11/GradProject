import segmentation_models_pytorch as smp
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Resize, ToTensor, Compose
import cv2
from pose_controler import pose_controller
from pose_errors import pose_errors
import threading
import math as mt
import socket
import signal
import struct
import sys
import time

HOST = '0.0.0.0'
PORT = 0

curve=0
world_glob = None
widthTop=100
heightTop=190
widthBottom=0
heightBottom=480
sleep = True
wT=640 
hT=480
curveList=[]
avgVal=5
points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop),
          (widthBottom , heightBottom ), (wT-widthBottom, heightBottom)])





def send_command(velocity, angle,conn):
    command = struct.pack('2B', int(velocity), int(angle)+60)
    conn.send(command)

def warpImg (img,points,w,h,inv=False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2,pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    return imgWarp
def getHistogram(img,display=False,minVal = 0.1,region=1):
 
    if region ==1:
          histValues = np.sum(img, axis=0)
    else :
          histValues = np.sum(img[img.shape[0]//region:,:], axis=0)   
    maxValue = np.max(histValues)  # FIND THE MAX VALUE
    minValue = minVal*maxValue
    indexArray =np.where(histValues >= minValue) # ALL INDICES WITH MIN VALUE OR ABOVE
    basePoint =  int(np.average(indexArray)) # AVERAGE ALL MAX INDICES VALUES
    
    if display:
        imgHist = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
        for x,intensity in enumerate(histValues):
           # print(intensity)
            if intensity > minValue:
                color=(255,0,255)
            else:
                color=(0,0,255)
            y2=int(img.shape[0]-(intensity//255//region)) 
            
            cv2.line(imgHist,(x,img.shape[0]),(x,y2),color,1)
            cv2.circle(imgHist,(basePoint,img.shape[0]),20,(0,255,255),cv2.FILLED)
        return basePoint,imgHist
    return basePoint
def display2(img,imgWarp,imgWarpPoints,imgHist,points,wT,hT,curve,display): 
    imgResult=img.copy()
    if display != 0:
        imgInvWarp = warpImg(imgWarp, points, wT, hT, inv=True)
        imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
        imgInvWarp[0:hT // 3, 0:wT] = 0, 0, 0
        imgLaneColor = np.zeros_like(img)
        imgLaneColor[:] = 0, 255, 0
        imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
        imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
        midY = 450
        cv2.putText(imgResult, str(round(curve,4)), (wT // 2 - 80, 85), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
        cv2.line(imgResult, (wT // 2, midY), (wT // 2 + (curve * 3), midY), (255, 0, 255), 5)
        cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY - 25), (wT // 2 + (curve * 3), midY + 25), (0, 255, 0), 5)
        for x in range(-30, 30):
            w = wT // 20
            cv2.line(imgResult, (w * x + int(curve // 50), midY - 10),
                     (w * x + int(curve // 50), midY + 10), (0, 0, 255), 2)
        #fps = cv2.getTickFrequency() / (cv2.getTickCount());
        #cv2.putText(imgResult, 'FPS ' + str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 50, 50), 3);
    if display == 2:
        imgStacked = stackImages(0.7, ([img, imgWarpPoints, imgWarp],
                                             [imgHist, imgLaneColor, imgResult]))
        return imgStacked
    elif display == 1:
        return imgResult
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver    
def drawPoints(img,points):
    for x in range( 0,4):
        cv2.circle(img,(int(points[x][0]),int(points[x][1])),15,(0,0,255),cv2.FILLED)
    return img
def LaneStay(frame,thresh):
    global curve
    frame = cv2.resize(frame, (640, 480))  # Resize the frame
    thresh=cv2.resize(thresh,(640,480))
    frame_withPoints=frame.copy()    
    # drawPoints(frame_withPoints,points)

    wrapped_img = warpImg(frame, points, wT, hT)
    wrapped_thresh_img=warpImg(thresh, points, wT, hT)
    
    middlePoint,imgHist1 = getHistogram(wrapped_thresh_img,True,region=4,minVal=0.5)
    curveAveragePoint,imgHist = getHistogram(wrapped_thresh_img, True, 0.9,1)
    curveRaw = curveAveragePoint-middlePoint

    curveRaw = curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)
    curve = int(sum(curveList)/len(curveList))
    
    # display_IMG=display2(frame,wrapped_thresh_img,frame_withPoints,imgHist,points,wT,hT,curve,2)
    
    return curve

def poly():
    global curve
    global sleep
    previous_steer_value=0
    model=None

    # Open the default camera (usually the webcam)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        try:

            # Capture frame-by-frame
            ret, frame = cap.read()

            # If frame is read correctly ret is True
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting...")
                break


            # Load the pre-trained model
            if model is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                model = smp.Unet(
                    encoder_name="mobilenet_v2",
                    encoder_weights="imagenet",
                    in_channels=3,
                    classes=1,
                ).to(device)
                model.load_state_dict(torch.load('afterLR (loss 273).pth',map_location=torch.device('cpu')))

            
            #Pre processing

            img_transform = Compose([
                Resize((96, 160)),  # Adjust this if needed after cropping
                ToTensor()
            ])
            
            # Load the saved model weights
        
            # image= cv2.imread(image_path)
            image= frame
                
            # Original dimensions
            height, width, _ = image.shape
            new_width = int(width * 1)
            new_height= int(height * 1)
            # Calculate the top left corner of the cropped frame
            start_x = (width - new_width) // 2
        
            
            # Crop the frame
            cropped_frame = image[ :,start_x:start_x + new_width]
            
                
            
            
            # Convert the cropped frame to PIL Image
            frame_pil = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
            
            

            
            # Apply transformations
            transformed_frame = img_transform(frame_pil).unsqueeze(0)
            transformed_frame = transformed_frame.to(device)
            
            
                    
            # Model prediction
            model.eval()
            with torch.no_grad():
                output = model(transformed_frame)
                output = torch.sigmoid(output[0])
            
            output = output.cpu()
            # Convert the binary mask back to an image
            binary_mask_image = Image.fromarray((output.numpy()[0] * 255).astype(np.uint8))
            
            binary_mask_resized = binary_mask_image.resize((new_width, height), Image.BILINEAR)
                
            # Create a new blank image with the same size as the original frame
            padded_mask = Image.new("L", (width, height))
                
            # # Calculate the position to paste the resized mask
            paste_x = (width - new_width) // 2
                
                
            # Paste the resized mask onto the blank image, centering it
            padded_mask.paste(binary_mask_resized, (paste_x,0))
            padded_mask= np.array(padded_mask)


            curve=LaneStay(image,padded_mask)
            print("error: ", curve)
            sleep = False



        except Exception as e:
            print("Error during image processing:", e)
            traceback.print_exc()

def check_pose_reached(errors, velocity, epsilon_p_max, v_p_max):
    return abs(errors[0]) <= epsilon_p_max and abs(-1 * mt.cos(errors[1]) * velocity) <= v_p_max


def control(epsilon_p_max, v_p_max):
    global curve
    global sleep
    uv = 0
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

            ## fix error 
            while True:
                errors, direction = pose_errors([0,0,0] , [0,0,curve/360 * (2*mt.pi)])
                
                print("errors:",errors)
                if check_pose_reached(errors, uv, epsilon_p_max, v_p_max):
                    print("centered")
                
                uv, udelta = pose_controller(errors, float(direction))
                
                print(f"Control Actions: {uv/max_velocity*100, udelta*360/(2*mt.pi)}\n")

                send_command(uv/max_velocity*100,udelta*360/(2*mt.pi),conn)

                while sleep:
                    time.sleep(0.1)
                sleep = True

if __name__ == '__main__':

    control_thread = threading.Thread(target = control, args= (2, 5))
    control_thread.start()
    poly()
