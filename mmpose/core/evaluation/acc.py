import numpy as np

from mmpose.core.post_processing import transform_preds


def _calc_distances(preds, targets, normalize):
    '''Calculate the normalized distances between preds and target.
    Note:
        batch_size: N
        num_keypoints: K

    Args:
        preds (np.ndarray[NxKx2]): Predicted keypoint location.
        targets (np.ndarray[NxKx2]): Groundtruth keypoint location.
        normalize (np.ndarray[Nx2]): Typical value is heatmap_size/10

    Returns:
        distances (np.ndarray[KxN]): The normalized distances.
        If target keypoints are missing, the distance is -1.
    '''
    N, K, _ = preds.shape
    distances = np.full((K, N), -1, dtype=np.float32)
    eps = np.finfo(np.float32).eps
    mask = (targets[..., 0] > eps) & (targets[..., 1] > eps)
    distances[mask.T] = np.linalg.norm(
        ((preds - targets) / normalize[:, None, :])[mask], axis=-1)
    return distances


def _distance_acc(distances, thr=0.5):
    '''Return the percentage below the distance threshold,
    while ignoring distances values with -1.

    Note:
        batch_size: N
    Args:
        distances (np.ndarray[N,]): The normalized distances.
        thr (float): Threshold of the distances.

    Returns:
        Percentage of distances below the threshold.
        If all target keypoints are missing, return -1.
    '''
    distance_valid = distances != -1
    num_distance_valid = distance_valid.sum()
    if num_distance_valid > 0:
        return (distances[distance_valid] < thr).sum() / num_distance_valid
    return -1


def _get_max_preds(heatmaps):
    '''Get keypoint predictions from score maps.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

    Returns:
        preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    '''
    assert isinstance(heatmaps, np.ndarray), \
        'heatmaps should be numpy.ndarray'
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    pred_mask = np.tile(maxvals > 0.0, (1, 1, 2))
    preds *= pred_mask
    return preds, maxvals


def pose_pck_accuracy(output, target, thr=0.5, normalize=None):
    '''Calculate the pose accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies

    Note:
        The PCK performance metric is the percentage of joints with
        predicted locations that are no further than a normalized
        distance of the ground truth. Here we use [w,h]/10.

        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        output (np.ndarray[N, K, H, W]): Model output heatmaps.
        target (np.ndarray[N, K, H, W]): Groundtruth heatmaps.
        thr
        normalize

    Returns:
        acc (np.ndarray[K]): Accuracy of each keypoint.
        avg_acc (float): Averaged accuracy across all keypoints.
        cnt (int): Number of valid keypoints.
    '''
    N, K, H, W = output.shape
    if K == 0:
        return None, 0, 0
    if normalize is None:
        normalize = np.tile(np.array([[H, W]]) / 10, (N, 1))

    pred, _ = _get_max_preds(output)
    gt, _ = _get_max_preds(target)
    distances = _calc_distances(pred, gt, normalize)

    acc = np.array([_distance_acc(d, thr) for d in distances])
    valid_acc = acc[acc >= 0]
    cnt = len(valid_acc)
    avg_acc = valid_acc.mean() if cnt > 0 else 0
    return acc, avg_acc, cnt


def get_final_preds(heatmaps, center, scale, post_process=True):
    """Get final keypoint predictions from heatmaps and transform them
    back to the image.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        center (np.ndarray[N, 2]): Center of the bounding box (x, y).
        scale (np.ndarray[N, 2]): Scale of the bounding box
            wrt height/width.
        post_process (bool): Option to use post processing or not.

    Returns:
        preds (np.ndarray[N, K, 2]): Predicted keypoint location in images.
        maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """

    coords, maxvals = _get_max_preds(heatmaps)

    heatmap_height = heatmaps.shape[2]
    heatmap_width = heatmaps.shape[3]

    if post_process:
        # add +/-0.25 shift to the predicted locations for higher acc.
        for n in range(coords.shape[0]):
            for k in range(coords.shape[1]):
                hm = heatmaps[n][k]
                px = int(coords[n][k][0])
                py = int(coords[n][k][1])
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = np.array([
                        hm[py][px + 1] - hm[py][px - 1],
                        hm[py + 1][px] - hm[py - 1][px]
                    ])
                    coords[n][k] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back to the image
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals
