import numpy as np

def interpolate_poses(start_pose, end_pose, num_intermediate_poses):
    """
    Generates a list of poses interpolated between two given poses.

    Parameters:
    ------------
    start_pose : list
        Starting pose in the format [x, y, theta].
    end_pose : list
        Ending pose in the format [x, y, theta].
    num_intermediate_poses : int
        Number of intermediate poses to generate.

    Returns:
    ------------
    list
        A list of poses including the start and end pose.
    """
    # Unpack start and end poses
    x0, y0, theta0 = start_pose
    x1, y1, theta1 = end_pose

    # Calculate step sizes
    x_steps = np.linspace(x0, x1, num_intermediate_poses + 2)
    y_steps = np.linspace(y0, y1, num_intermediate_poses + 2)
    theta_steps = np.linspace(theta0, theta1, num_intermediate_poses + 2)

    # Generate poses
    poses = [[x_steps[i], y_steps[i], theta_steps[i]] for i in range(num_intermediate_poses + 2)]
    return poses

# Example usage
start_pose = [6.9, 3.42, 0.36]
end_pose = [2.15, 1.15, 0]
poses = interpolate_poses(start_pose, end_pose, 3) 
print(poses)
