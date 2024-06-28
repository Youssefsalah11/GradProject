import math
def pose_controller(qp, d):
    """
    Description:
    ------------
    The `pose_controller` function calculates the control inputs needed to drive a vehicle
    to a desired pose based on the error vector and direction of motion.

    Parameters:
    ------------
    tuple
        A tuple containing:
        - qp : list of floats
            An error vector in polar coordinates in the form [ep, ealpha, ebeta], where:
            - ep : float : Error in position.
            - ealpha : float : Alignment to the target position.
            - ebeta : float : Alignment with the desired orientation.
        - d : float
            Direction of motion, which is positive when the desired pose is in front of the car 
            and negative when it is behind.

    Returns:
    ------------
    u : tuple of two floats
        The control inputs for the vehicle to reach the desired pose. The control law is defined as:
        - Uv : float : Control of longitudinal motion of the vehicle with velocity RvRx.
        - Udelta : float : Turning angle of the front steerable wheel δ.

    Raises:
    ------------
    ValueError
        If the input qp is not a list of three floats or d is not a float.

    Examples:
    ------------
    Example usage of the function, including input parameters and expected output.
    
    Example:
    --------
    >>> qp = [2.8284271247461903, 0.7853981633974483, 0.21460183660255172]
    >>> d = 1.0
    >>> u = pose_controller(qp, d)
    >>> print(u)
    (5.656854249492381, 1.2853981633974483)
    """

    if not (isinstance(qp, list) and len(qp) == 3 and all(isinstance(val, float) for val in qp)):
        raise ValueError("qp must be a list of three floats: [ep, ealpha, ebeta].")
    if not isinstance(d, float):
        raise ValueError("d must be a float.")

    # Unpack the error vector
    ep, ealpha, ebeta = qp

    # Control gains (these should be tuned for the specific application)
    kp = 6     # Gain for position error
    kalpha = 9   # Gain for alignment error
    kbeta = -4    # Gain for orientation error
    l = 1 # Wheelbase length in meters

    # Constraints (these values should be set according to vehicle specifications)
    max_velocity = 16   # Maximum linear velocity (m/s)
    max_acceleration = 9.8/6    # Maximum linear acceleration (m/s^2)
    max_turn_angle = math.pi/6      # Maximum turning angle (radians)
    max_turn_rate = max_velocity * math.tan(max_turn_angle) / l  # Maximum angular velocity of turning (rad/s)

    # Control law
    Uv = kp * ep
    Udelta = kalpha * ealpha + kbeta * ebeta

    # Adjust control inputs based on direction of motion
    Uv *= d
    Udelta *= d

    # Apply constraints to control inputs
    Uv = max(-max_velocity, min(Uv, max_velocity))
    Udelta = max(-max_turn_angle, min(Udelta, max_turn_angle))

    return (Uv, Udelta)
