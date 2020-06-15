import numpy as np


def affine_transform(pt, trans):
    """Apply an affine transformation to the points.

    Args:
        pt (np.ndarray): 2 dim point to be transformed
        trans (np.ndarray): 2x3 matrix of an affine transform

    Returns:
        Transformed points.
    """
    new_pt = np.matrix(trans) * np.matrix([pt[0], pt[1], 1.]).T
    new_pt = np.array(new_pt)[:, 0]
    return new_pt


def get_3rd_point(a, b):
    """Get 3rd point from point a and point b.

    The 3rd point is defined by rotate point a 90 degrees anticlockwise,
    using b as rotation center.

    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)

    Returns:
        The 3rd point is defined by rotate point a 90 degrees anticlockwise,
    using b as rotation center.
    """
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    """Rotate the vector by an angle

    Args:
        src_point (list[float]): source point
        rot_rad (float): rotation angle by radius

    Returns:
        Rotation matrix
    """

    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result
