import segmentation_models_pytorch as smp
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Resize, ToTensor, Compose
import cv2
import math as mt
import struct
import sympy as sp
from PIL import Image
import cv2
import torchvision.transforms as T
from flask import Flask, request, jsonify
import io


model = None
app = Flask(__name__)
device = None

def load_image(image):
    transforms = T.Compose([
        T.ToTensor(),
    ])
    return image


def poly(image):

    #Pre processing
    img_transform = Compose([
        Resize((96, 160)),  # Adjust this if needed after cropping
        ToTensor()
    ])
#     print(image)

    # image= cv2.imread(image_path)
    image2 = image
        
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
        output = torch.sigmoid(output[0]).cpu()
        
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
    
    # plt.figure(figsize=(6, 6))
    # plt.title("label")
    # plt.imshow(padded_mask)  # Display the image in grayscale
    # plt.show()
    high_thresh, thresh_im = cv2.threshold(padded_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    lowThresh = 0.5*high_thresh
    resized_mask = cv2.resize(padded_mask, (160,96), interpolation=cv2.INTER_AREA)
    
    # Define the Gaussian blur parameters
    kernel_size = (21,21)  # Example kernel size, can be adjusted
    sigma_x = 0  # Standard deviation in X; if 0, it is calculated from the kernel size
    
    # Apply Gaussian blur
    blurred_mask = cv2.GaussianBlur(resized_mask, kernel_size, sigma_x)

    
    edges = cv2.Canny(blurred_mask, lowThresh, high_thresh)
    
    # Perform Hough Line Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=10, minLineLength=5, maxLineGap=10)
    # lines = cv2.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,threshold = 20,minLineLength = 2,maxLineGap = 4)
    
    
    
    # plt.figure(figsize=(6, 6))
    # plt.title("label")
    # plt.imshow(edges, cmap='gray')  # Display the image in grayscale
    # plt.show()
    
    def get_polynomials ():
    
        def extract_poly(window , image):
    
            xy = []
            for i in range(len(window)):
                col = window[i]
                j = find_first(255, col)
                if j != -1:
                    xy.extend((i, j))
            
            # Reshape into [[x1, y1],...]
            data = np.array(xy).reshape((-1, 2))
            
            # Translate points back to original positions.
            data[:, 1] = bounds[1] - data[:, 1]
            
            
            xdata = data[:,0]
            ydata = data[:,1]
            
            try:
                z = np.polyfit(ydata, xdata, 2)
            except:
                z = 0

            return z, [min(xdata),max(xdata)]
       
        # Check if any lines were found
        left_lines = []
        right_lines = []
        image_with_left_lines= np.zeros((edges.shape[0],edges.shape[1], 1), dtype=np.uint8) 
        image_with_right_lines= np.zeros((edges.shape[0],edges.shape[1], 1), dtype=np.uint8) 
    
        if lines is not None:
            
            for line in lines:
                    x1, y1, x2, y2 = line[0]  # Extract line endpoints
            
                    # Calculate the slope of the line
                    if x2 - x1 != 0: 
                        slope = (y2 - y1) / (x2 - x1)
                        
                        # Define a slope threshold to identify vertical lines
                        slope_threshold = 0.5  
            
                        # Check if the line is not too close to horizontal
                        if abs(slope) > slope_threshold:
                            # Determine if it's a left or right line based on the x-coordinate of the midpoint
                            midpoint_x = (x1 + x2) / 2
            
                            # Assuming the image width is 'img_width' (you should replace this with the actual image width)
                            img_width = edges.shape[1]
            
                            if midpoint_x < img_width / 2:
                                left_lines.append(line)
                                
                            else:
                                right_lines.append(line)
        else: return "lines are not clear"
        
        for line in left_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_with_left_lines, (x1, y1), (x2, y2), (255, 255, 0), 1)
        
        
        for line in right_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_with_right_lines, (x1, y1), (x2, y2), (255, 255, 0), 1)
        
    
        def find_first(item, vec):
            """return the index of the first occurence of item in vec"""
            for i in range(len(vec)):
                if item == vec[i]:
                    return i
            return -1
        
        
        bounds = [0,96]
        bounds2 = [0,160]
        
        window = image_with_left_lines[bounds[1]:bounds[0]:-1].transpose()[bounds2[0]:bounds2[1]:1][0]
        left_line_poly = extract_poly(window , image_with_left_lines)
        
        
        window = image_with_right_lines[bounds[1]:bounds[0]:-1].transpose()[bounds2[0]:bounds2[1]:1][0]
        right_line_poly = extract_poly(window, image_with_right_lines)

        ##left line
        fL = np.poly1d(left_line_poly[0])
        # Generate x values for plotting the polynomial curve
        tL = np.arange(60,96, 1)

        ##right line
        fR = np.poly1d(right_line_poly[0])
        # Generate x values for plotting the polynomial curve
        tR = np.arange(60,96, 1)
        

        
        # Assuming image2 is defined and is a numpy array
        height2, width2 = image2.shape[:2]
        
        # Scaling factors
        scale_x = width2 / 160
        scale_y = height2 / 96
        
        # Function to generate scaled points for the polynomial
        def generate_scaled_poly_points(func, t_values, scale_x, scale_y):
            return np.array([[func(t) * scale_x, t * scale_y] for t in t_values], np.int32)
        
        # Convert grayscale image to BGR for color drawing
        if len(image2.shape) == 2:
            image2_color = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
        else:
            image2_color = image2.copy()
        
        # Generate scaled points for the left and right polynomials
        pointsL = generate_scaled_poly_points(fL, tL, scale_x, scale_y)
        pointsR = generate_scaled_poly_points(fR, tR, scale_x, scale_y)
        
        # Draw the polynomial lines
        cv2.polylines(image2_color, [pointsL], False, (255, 0, 0), 5) 
        cv2.polylines(image2_color, [pointsR], False, (255, 0, 0), 5) 

        # Draw a vertical line at the center of the image
        center_x = width2 // 2
        cv2.line(image2_color, (center_x, 0), (center_x, int(height2*0.8)), (0, 0,255), 2)  # Green color for the vertical line
        
        # Function to calculate midpoints between two sets of points
        def calculate_midpoints(points1, points2):
            midpoints = []
            for p1, p2 in zip(points1, points2):
                mid_x = (p1[0] + p2[0]) // 2
                mid_y = (p1[1] + p2[1]) // 2
                midpoints.append([mid_x, mid_y])
            return np.array(midpoints, np.int32)
        
        # Calculate midpoints and draw the center line
        center_points = calculate_midpoints(pointsL, pointsR)
        # Filter center_points to include only those with y-values greater than height2 * 0.9
        filtered_center_points = np.array([point for point in center_points if point[1] > height2 * 0.8], dtype=np.int32)
        

        # Calculate the average x-coordinate of the filtered points
        if len(filtered_center_points) > 0:
            average_x = int(np.mean(filtered_center_points[:, 0]))
        
            # Draw a perfectly vertical line using the average x-coordinate
            cv2.line(image2_color, (average_x, int(height2 * 0.8)), (average_x, height2), (255, 0, 0), 5)

        # Find the y-coordinate at the center of the image (or at the relevant y-level)
        center_y = height2 // 2
        
        # Function to find the closest point on a line to a given y-coordinate
        def find_closest_point(points, target_y):
            closest_point = min(points, key=lambda point: abs(point[1] - target_y))
            return closest_point
        
        # Find the closest points
        closest_points = find_closest_point(filtered_center_points, center_y)

        
        # Draw horizontal lines from these points to the center line
        cv2.line(image2_color, (closest_points[0], center_y), (center_x, center_y), (0, 0, 255), 5)
        cv2.line(image2_color, (closest_points[0]+1,center_y-10),(closest_points[0]+1,center_y+10),(0, 0, 255), 10)
        cv2.line(image2_color, (center_x+1,center_y-10),(center_x+1,center_y+10),(0, 0, 255), 10)


        return right_line_poly , left_line_poly, image2_color

    
    def calculate_steering_angle(right_poly, left_poly, image_width,image_color,vehicle_center=None):
        """
        Calculate the steering angle based on the polynomials of the lane lines.
    
        Args:
        - right_poly (np.poly1d): Polynomial for the right lane line.
        - left_poly (np.poly1d): Polynomial for the left lane line.
        - image_width (int): Width of the camera image.
        - vehicle_center (int, optional): X-coordinate of the vehicle center. Defaults to middle of the image.
    
        Returns:
        - float: Required steering angle.
        """

        vehicle_center = image_width / 2
    
        # Distance from the vehicle (bottom of the image)
        y_pos = 0
        
        x = sp.symbols('x')
        

        lines_average = (left_line_poly_tuple[0]+right_line_poly_tuple[0])/2
        a, b, c = lines_average

        # Define the quadratic function
        f_x = a * x**2 + b * x + c
        
        # Calculate the first derivative of f(x)
        f_prime = sp.diff(f_x, x)
        
        # Calculate the second derivative of f(x)
        f_double_prime = sp.diff(f_prime, x)
        
        # Define the curvature formula
        curvature = abs(f_double_prime) / (1 + f_prime**2)**(3/2)
        # print(f_double_prime)
        y_pos = 96
        #steering angle intialized to 0
        steering_angle = 0
        for y_pos in range(y_pos,70, -5):
           # Calculate x positions of lane lines at y_pos
            x_left = left_poly(y_pos)
            x_right = right_poly(y_pos)
            # Calculate lane center
            lane_center = (x_left + x_right) / 2
            
            # Calculate deviation
            deviation = vehicle_center - lane_center
            #deviation +=curvature.subs(x, y_pos)*30
        
            # Steering angle proportional to deviation
            Kp = 0.10  # Proportional gain, adjust as necessary
            steering_angle += Kp * deviation
    
        return steering_angle


    
    # print(get_polynomials())
    right_line_poly_tuple, left_line_poly_tuple , image_color=get_polynomials()
    image_width = 160
    
    # Extract np.poly1d objects from tuples
    right_line_poly = np.poly1d(right_line_poly_tuple[0])
    left_line_poly = np.poly1d(left_line_poly_tuple[0])
    steering_angle = calculate_steering_angle(left_line_poly, right_line_poly, image_width, image_color)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # if steering_angle > 15.0:
    #     instruction = "Sharp Right"
    # elif 8 < steering_angle <= 15.0:
    #     instruction = "Turn Right"
    # elif 5 < steering_angle <= 8:
    #     instruction = "Slight Right"
    # elif -5 <= steering_angle <= 5:
    #     instruction = "Go Straight"
    # elif -6 <= steering_angle < -6:
    #     instruction = "Slight Left"
    # elif -15.0 <= steering_angle < -5:
    #     instruction = "Turn Left"
    # else:  # steering_angle < -1.0
    #     instruction = "Sharp Left"
    
    # # Display the steering instruction
    # cv2.putText(image_color, instruction, (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Display the steering angle
    # cv2.putText(image_color, f"Steering Angle: {steering_angle:.2f}", (10, 90), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    
    # Display the image
    # cv2.imshow('Polynomial Overlay', image_color)
    cv2.waitKey(1)
    # cv2.destroyAllWindows()
    #Steering Angle Value:
    # The steering angle is usually measured in degrees.
    # A positive value often indicates that the steering should turn to the left.
    # A negative value typically indicates that the steering should turn to the right.
    # A value of zero means the steering should remain straight, implying the vehicle is correctly aligned with the desired path or lane.
    return steering_angle
    





@app.route('/call_function', methods=['POST'])
def call_function():
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert("L")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    angle = poly(image)
        
    # Return the result as JSON
    return jsonify({'result': angle})


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)
    model.load_state_dict(torch.load('../Model Weights/afterLR (loss 273).pth',map_location=torch.device('cpu')))
    app.run(host='0.0.0.0', port=5003)
