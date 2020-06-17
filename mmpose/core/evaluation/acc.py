import numpy as np

from mmpose.core.post_processing import transform_preds


def _calc_distances(preds, targets, normalize):
    '''Calculate the normalized distances between preds and target.
    Note:
        batch_size: N (n)
        num_keypoints: K (k)

    Args:
        preds (np.ndarray[NxKx2]): Predicted keypoint location.
        target (np.ndarray[NxKx2]): Groundtruth keypoint location.
        normalize (np.ndarray[Nx2]): Normalization factor (heatmap_size/10).

    Returns:
        distances (np.ndarray[KxN]): The normalized distances.
        If target keypoints are missing, the distance is -1.
    '''

    distances = np.zeros((preds.shape[1], preds.shape[0]))
    eps = np.finfo(np.float32).eps
    for n in range(preds.shape[0]):
        for k in range(preds.shape[1]):
            if targets[n, k, 0] > eps and targets[n, k, 1] > eps:
                normed_preds = preds[n, k, :] / normalize[n]
                normed_targets = targets[n, k, :] / normalize[n]
                distances[k, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                distances[k, n] = -1
    return distances


def _distance_acc(distances, thr=0.5):
    '''Return the percentage below the distance threshold,
    while ignoring distances values with -1.

    Note:
        batch_size: N (n)
    Args:
        distances (np.ndarray[N,]): The normalized distances.
        thr (float): Threshold of the distances.

    Returns:
        Percentage of distances below the threshold.
        If all target keypoints are missing, return -1.
    '''
    distance_valid = np.not_equal(distances, -1)
    num_distance_valid = distance_valid.sum()
    if num_distance_valid > 0:
        return np.less(distances[distance_valid],
                       thr).sum() / num_distance_valid
    else:
        return -1


def _get_max_preds(heatmaps):
    '''Get keypoint predictions from score maps.

    Note:
        batch_size: N (n)
        num_keypoints: K (k)
        heatmap height: h
        heatmap width: w

    Args:
        heatmaps (np.ndarray[N, K, h, w]): model predicted heatmaps.

    Returns:
        preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    '''
    assert isinstance(heatmaps, np.ndarray), \
        'heatmaps should be numpy.ndarray'
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    width = heatmaps.shape[3]
    heatmaps_reshaped = heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % width
    preds[:, :, 1] = preds[:, :, 1] // width

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))

    preds *= pred_mask
    return preds, maxvals


def pose_pck_accuracy(output, target, thr=0.5):
    '''Calculate the pose accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies

    Note:
        The PCK performance metric is the percentage of joints with
        predicted locations that are no further than a normalized
        distance of the ground truth. Here we use [w,h]/10.

        batch_size: N (n)
        num_keypoints: K (k)
        heatmap height: h
        heatmap width: w

    Args:
        output (np.ndarray[N, K, h, w]): Model output heatmaps.
        target (np.ndarray[N, K, h, w]): Groundtruth heatmaps.

    Returns:
        acc (np.ndarray[K]): Accuracy of each keypoint.
        avg_acc (float): Averaged accuracy across all keypoints.
        cnt (int): Number of valid keypoints.
    '''
    if output.shape[1] == 0:
        return None, 0, 0

    idx = list(range(output.shape[1]))
    pred, _ = _get_max_preds(output)
    gt, _ = _get_max_preds(target)
    h = output.shape[2]
    w = output.shape[3]

    norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10

    distances = _calc_distances(pred, gt, norm)

    acc = np.zeros(len(idx))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i] = _distance_acc(distances[idx[i]], thr)
        if acc[i] >= 0:
            avg_acc = avg_acc + acc[i]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    return acc, avg_acc, cnt


def get_final_preds(heatmaps, center, scale, post_process=True):
    """Get final keypoint predictions from heatmaps and transform them
    back to the image.

    Note:
        batch_size: N (n)
        num_keypoints: K (k)
        heatmap height: h
        heatmap width: w

    Args:
        heatmaps (np.ndarray[N, K, h, w]): model predicted heatmaps.
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
