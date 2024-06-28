import math

def pose_errors(pose_data, position):
    """
    Description:
    ------------
    This function is called after `desired_motion` and takes the input desired pose 
    and car kinematics, then calculates the errors for the pose controller (next step).

    Parameters:
    ------------
    pose_data : list of floats
        The desired pose in the format [0xd, 0yd, 0phid], where:
        - 0xd : float : Desired x-coordinate.
        - 0yd : float : Desired y-coordinate.
        - 0phid : float : Desired orientation angle in radians.

    position : list of floats
        The current position of the vehicle in the format [0xR, 0yR, 0phiZ], where:
        - 0xR : float : Current x-coordinate.
        - 0yR : float : Current y-coordinate.
        - 0phiZ : float : Current orientation angle in radians.

    Returns:
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

    Raises:
    ------------
    ValueError
        If the input pose_data or position do not match the expected format or contain invalid values.

    Examples:
    ------------
    Provide example usage of the function, including input parameters and expected 
    output. This helps to illustrate how the function should be used.

    Example:
    --------
    >>> pose_data = [3.0, 4.0, 1.0]
    >>> position = [1.0, 2.0, 0.5]
    >>> qp, d = pose_errors(pose_data, position)
    >>> print(qp)
    [2.8284271247461903, 0.7853981633974483, 0.21460183660255172]
    >>> print(d)
    1.0

    """

    if not (isinstance(pose_data, list) and len(pose_data) == 3):
        raise ValueError("pose_data must be a list of three floats: [0xd, 0yd, 0phid].")
    if not (isinstance(position, list) and len(position) == 3):
        raise ValueError("position must be a list of three floats: [0xR, 0yR, 0phiZ].")

    # Unpack the pose data
    xd, yd, phid = pose_data
    xR, yR, phiZ = position

    # Calculate the position error (ep)
    ep = math.sqrt((xd - xR) ** 2 + (yd - yR) ** 2)

    # Calculate the direction of motion (d)
    dx = xd - xR
    dy = yd - yR
    alpha = math.atan2(dy, dx) - phiZ
    d = 1 if (abs(alpha) <= (math.pi/2)) else -1

    # Just a helper function
    sign = lambda x: math.copysign(1, x)

    epsi = phid - phiZ
    
    # Calculate the angular direction error (ealpha)
    ealpha = alpha if (d == 1) else (alpha - math.pi * sign(alpha))

    # Calculate the course angle error (ebeta)
    ebeta = epsi - ealpha

    # Normalize angles to be within the range [-pi, pi]
    # ealpha = (ealpha + math.pi) % (2 * math.pi) - math.pi
    # ebeta = (ebeta + math.pi) % (2 * math.pi) - math.pi

    # Error vector in polar coordinates
    qp = [ep, ealpha, ebeta]

    return qp, d
