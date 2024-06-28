import numpy as np

def kinematic_model(u, current_pose, current_velocity, current_angular_velocity, dt):
    """
    Description:
    ------------
    The `kinematic_model` function calculates the actual pose and other motion parameters of the
    vehicle based on the kinematic model of a car.

    Parameters:
    ------------
    u : tuple of two floats
        The control inputs for the vehicle. The control law is defined as:
        - Uv : float : Control of longitudinal motion of the vehicle with velocity RvRx.
        - Udelta : float : Turning angle of the front steerable wheel δ.
    current_pose : tuple of three floats
        The current pose of the vehicle in the form (x, y, theta), where:
        - x : float : Current x-coordinate.
        - y : float : Current y-coordinate.
        - theta : float : Current orientation angle in radians.
    current_velocity : float
        Current linear velocity of the vehicle (m/s).
    current_angular_velocity : float
        Current angular velocity of the vehicle (rad/s).
    dt : float
        Time step for the simulation (s).

    Returns:
    ------------
    tuple
        The updated pose of the vehicle in the form (x, y, theta), where:
        - x : float : Updated x-coordinate.
        - y : float : Updated y-coordinate.
        - theta : float : Updated orientation angle in radians.
    float
        Updated turning angle of the front steerable wheel δ.
    float
        Updated Rωz.
    float
        Updated RaRx.
    """

    # Unpack control inputs
    Uv, Udelta = u

    # Update pose based on kinematic model
    x, y, theta = current_pose
    x += current_velocity * np.cos(theta) * dt
    y += current_velocity * np.sin(theta) * dt
    theta += current_velocity / RaRx * np.tan(Udelta) * dt

    # Limit theta to range [-pi, pi]
    theta = np.mod(theta + np.pi, 2*np.pi) - np.pi

    # Update turning angle of the front steerable wheel δ
    delta = Udelta

    # Update Rωz
    R_omega_z = current_velocity / RaRx * np.tan(Udelta)

    # Return updated pose and motion parameters
    return (x, y, theta), delta, R_omega_z, RaRx

# Example usage
initial_pose = (0.0, 0.0, 0.0)  # Initial pose (x, y, theta) of the vehicle
current_velocity = 1.0          # Current linear velocity of the vehicle (m/s)
current_angular_velocity = 0.0  # Current angular velocity of the vehicle (rad/s)
dt = 0.1                         # Time step for simulation (s)

# Control inputs (Uv, Udelta)
u = (1.0, np.pi/4)  # Move forward at 1 m/s and turn 45 degrees

# Vehicle parameters
RaRx = 1.0  # Assuming a value for RaRx

# Calculate updated pose and motion parameters
updated_pose, delta, R_omega_z, _ = kinematic_model(u, initial_pose, current_velocity, current_angular_velocity, dt)

# Print the updated pose and motion parameters
print("Updated Pose:", updated_pose)
print("Turning Angle δ:", delta)
print("Rωz:", R_omega_z)
print("RaRx:", RaRx)
