import numpy as np
from numpy.testing import assert_array_almost_equal

from mmpose.core.post_processing.shared_transforms import (affine_transform,
                                                           get_3rd_point,
                                                           get_dir)


def test_get_3rd_point():
    a = np.array([0, 1])
    b = np.array([0, 0])
    assert_array_almost_equal(
        get_3rd_point(a, b), np.array([-1, 0]), decimal=4)


def test_affine_transform():
    pt = np.array([0, 1])
    trans = np.array([[1, 0, 1], [0, 1, 0]])
    assert_array_almost_equal(
        affine_transform(pt, trans), np.array([1, 1]), decimal=4)


def test_get_dir():
    src_point = np.array([0, 1])
    rot_rad = np.pi / 2
    assert_array_almost_equal(
        get_dir(src_point, rot_rad), np.array([-1, 0]), decimal=4)
