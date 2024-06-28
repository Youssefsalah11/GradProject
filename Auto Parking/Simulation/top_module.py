import numpy as np
import math as mt
from box_finder import box_finder
from desired_motion import desired_motion, initialize_poses
from slots_data_finder import slots_data_finder
from pose_controler import pose_controller
from pose_errors import pose_errors
from update_vehicle import update_vehicle_pose
import matplotlib.pyplot as plt
import requests

def check_pose_reached(errors, velocity, epsilon_p_max, v_p_max):
    return abs(errors[0]) <= epsilon_p_max and abs(-1 * mt.cos(errors[1]) * velocity) <= v_p_max

def main_simulation_loop(initial_position, dt, epsilon_p_max, v_p_max):
    current_position = list(initial_position)
    
    # Initialize the plot
    # plt.figure(figsize=(10, 10))
    # plt.axis('equal')
    # plt.gca().invert_yaxis()
    # plt.grid(True)
    
    # Just an example of selecting pose ids in a sequential manner
    for pose_id in range(len(poses_data)):
        count = 0
        uv =0
        desired_pose = desired_motion(pose_id)

        while True:
            count += 1
            errors, direction = pose_errors(desired_pose, current_position)
            if check_pose_reached(errors, uv, epsilon_p_max, v_p_max):
                break
            
            uv, udelta = pose_controller(errors, float(direction))
            current_position = update_vehicle_pose((uv, udelta), current_position, dt)
            
            # Plot current and desired positions
            plt.plot([current_position[0]], [current_position[1]], 'bo')  # current position in blue
            plt.plot([desired_pose[0]], [desired_pose[1]], 'rx')  # desired position in red
            plt.pause(0.05)  # pause to update the plot
            
            # Print the current state for monitoring
            print(f"Step {count}:")
            print(f"Current Pose: {current_position}")
            print(f"Desired Pose: {desired_pose}")
            print(f"Control Actions: {uv, udelta}\n")
        
        # Draw a line to the next pose
        # plt.plot([prev_position[0], current_position[0]], [prev_position[1], current_position[1]], 'k-')
        # prev_position = current_position
        plt.pause(0.05)  # pause to update the plot

    plt.show()


def call_server_function(param):
    url = 'http://localhost:5000/call_function'
    payload = {'param': param}
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        return result['result']
    else:
        return f"Error: {response.status_code}"

# Example usage
result = call_server_function('example parameter')
print(result)


# # Example data setup
# poses_data = [
#     [6.9, 3.42,0.36],
#     [6.45, 3.25, 0.46],
#     [3.1,1.5, 0.46],
#     [1.1, 1.15, 0],
#     [2.15, 1.15, 0]
# ]


# mid_point = [(1.1+12)/2, (1.15+3.8)/2,  0.4]

# mid_mid_point1 = [(mid_point[0]+12)/2,(mid_point[1]+3.8)/2,0.3]
# mid_mid_point2 = [(mid_point[0]+1.1)/2,(mid_point[1]+1.15)/2,0.3]
# poses_data = [
#     # mid_mid_point1,
#     mid_point,
#     # mid_mid_point2,
#     [1.1, 1.15, 0], #endpoint
    
# ]

prediction = box_finder("occupancy_grid_raw_image.png")



poses_data = slots_data_finder(prediction)


# poses_data = [{'end': [43, 69, 1.62],
#                 'mid': [52, 52,  2.355], 
#                 'start': [59, 36, 1.57], 
#                 'end2':[43,65,1.57]}]
if len(poses_data) == 0:
    raise ("no poses found")



initialize_poses(poses_data)
# print(poses_data)
# initial_position = (12, 4.5, 0.0)
initial_position = poses_data[0]['start']

poses_data = [poses_data[0]['mid'],poses_data[0]['end']]

main_simulation_loop(initial_position, 0.03, 0.5, 0.5)
