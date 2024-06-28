poses_data = []

def initialize_poses(data):
    """
    Initialize the global poses_data variable with the provided data.

    Parameters:
    ------------
    data : list of lists
        A list containing multiple pose arrays. Each pose array includes the pose data, 
        such as [0xd, 0yd, 0phid]. This data is initialized once at the start and represents 
        different poses (e.g., 0p1d, 0p2d, ..., 0pnd).
    """
    global poses_data
    poses_data = [data[0]['mid'],data[0]['end1'],data[0]['end2']]

def desired_motion(pose_id):
    """
    The `desired_motion` function outputs the desired pose from a set of predefined poses. 
    It takes a specific pose identifier from the pose selector to retrieve the corresponding desired pose.

    Parameters:
    ------------
    pose_id : int
        An identifier from the pose selector used to select the desired pose from `poses_data`.
        It should be a valid index within the range of `poses_data`.

    Returns:
    ------------
    list
        A list containing the desired motion data corresponding to the given `pose_id`. The format 
        of the returned list is [0xd, 0yd, 0phid], where:
        - 0xd : float : Desired x-coordinate.
        - 0yd : float : Desired y-coordinate.
        - 0phid : float : Desired orientation angle in radians.

    Raises:
    ------------
    IndexError
        If `pose_id` is out of the range of `poses_data`.
    TypeError
        If `pose_id` is not an integer.

    Examples:
    ------------
    Example usage of the function:

    >>> initialize_poses([
    ...     [1.0, 2.0, 0.5],
    ...     [2.0, 3.0, 1.0],
    ...     [3.0, 4.0, 1.5]
    ... ])
    >>> pose_id = 1
    >>> result = desired_motion(pose_id)
    >>> print(result)
    [2.0, 3.0, 1.0]

    """

    global poses_data

    if not isinstance(pose_id, int):
        raise TypeError("pose_id must be an integer.")
    if pose_id < 0 or pose_id >= len(poses_data):
        raise IndexError("pose_id is out of range.")

    return poses_data[pose_id]
