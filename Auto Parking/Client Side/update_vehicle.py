import numpy as np
import math
def update_vehicle_pose(u, current_pose, dt):
    """
    Update the vehicle pose based on control inputs using a simple kinematic bicycle model.

    Parameters:
    ------------
    u : tuple of two floats
        The control inputs for the vehicle:
        - Uv : float : Control of longitudinal motion of the vehicle with velocity RvRx.
        - Udelta : float : Turning angle of the front steerable wheel δ.
    current_pose : tuple of three floats
        The current pose of the vehicle in the form (x, y, theta).
    dt : float
        Time step for the simulation.

    Returns:
    ------------
    list
        The updated pose of the vehicle in the form (x, y, theta).
    """
    Uv, Udelta = u
    x, y, theta = current_pose
    l = 2.9
    # If the steering angle is effectively zero, simplify the calculations for straight line motion.
    if abs(Udelta) < 1e-6:  # Threshold for considering the steering angle as zero
        new_x = x + Uv * np.cos(theta) * dt
        new_y = y + Uv * np.sin(theta) * dt
        new_theta = theta # No change in orientation if steering angle is zero
    else:

        # Update the orientation of the vehicle
        new_theta = theta + math.tan(Udelta) * Uv / l * dt

        # Calculate the new position of the vehicle
        new_x = x + Uv * np.cos(theta) * dt
        new_y = y + Uv * np.sin(theta) * dt

    return [new_x, new_y, new_theta]
