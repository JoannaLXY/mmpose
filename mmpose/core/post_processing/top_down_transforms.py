# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

import cv2
import numpy as np

from .shared_transforms import affine_transform, get_3rd_point, rotate_point


def fliplr_joints(joints_3d, joints_3d_visible, img_width, flip_pairs):
    """Flip human joints horizontally.

    Note:
        num_keypoints: K (k)

    Args:
        joints_3d (np.ndarray([K, 3])): Coordinates of keypoints.
        joints_3d_visible (np.ndarray([K, 1])): Visibility of keypoints.
        img_width (int): Image width.
        flip_pairs (list[tuple()): Pairs of keypoints which are mirrored
            (for example, left ear -- right ear).
    Returns:
        flipped joints_3d & joints_3d_visible
    """

    assert len(joints_3d) == len(joints_3d_visible)
    assert img_width > 0

    # Flip horizontal
    joints_3d[:, 0] = img_width - 1 - joints_3d[:, 0]

    # Change left-right parts
    for pair in flip_pairs:
        joints_3d[pair[0], :], joints_3d[pair[1], :] = \
            joints_3d[pair[1], :], joints_3d[pair[0], :].copy()
        joints_3d_visible[pair[0], :], joints_3d_visible[pair[1], :] = \
            joints_3d_visible[pair[1], :], joints_3d_visible[pair[0], :].copy()

    return joints_3d * joints_3d_visible, joints_3d_visible


def flip_back(output_flipped, flip_pairs):
    """Flip the flipped heatmaps back to the original form.

    Note:
        batch_size: N (n)
        num_keypoints: K (k)
        heatmap height: h
        heatmap width: w

    Args:
        ouput_flipped (np.ndarray[N, K, h, w]): The output heatmaps obtained
            from the flipped images.
        flip_pairs (list[tuple()): Pairs of keypoints which are mirrored
            (for example, left ear -- right ear).
    Returns:
        ouput_flipped: the heatmaps are flipped back to the original images.
    """
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_keypoints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]
    for pair in flip_pairs:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp
    return output_flipped


def transform_preds(coords, center, scale, output_size):
    """Get final keypoint predictions from heatmaps and transform them
    back to the image.

    First calculate the trans matrix from _get_affine_transform(),
    then affine transform the src coords to the dst coords.

    Note:
        num_keypoints: K (k)
    Args:
        coords (np.ndarray[K, 2]): Predicted keypoint location.
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt height/width.
        output_size (np.ndarray[2, ]): Size of the destination heatmaps.

    Returns:
        target_coords: predicted coordinates in the images.
    """
    assert coords.shape[1] == 2
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=False):
    """Get the affine transform matrix, given
    center/scale/rot/output_size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt height/width.
        rot (float): Rotation factor.
        output_size (np.ndarray[2, ]): Size of the destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default np.array([0, 0].
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        trans: the transform matrix.
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans
