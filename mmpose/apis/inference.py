import os

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmpose.datasets.pipelines import Compose
from mmpose.models import build_posenet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def init_pose_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a pose model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_posenet(config.model)
    if checkpoint is not None:
        # load model checkpoint
        load_checkpoint(model, checkpoint, map_location=device)
    # save the config in the model for convenience
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def box2cs(cfg, box):
    """This encodes bbox(x,y,w,h) into (center, scale)

    Args:
        x, y, w, h

    Returns:
        center (np.ndarray[float32](2,)): center of the bbox (x, y).
        scale (np.ndarray[float32](2,)): scale of the bbox w & h.
    """

    x, y, w, h = box[:4]
    input_size = cfg.data_cfg['image_size']
    aspect_ratio = input_size[0] / input_size[1]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    # pixel std is 200.
    scale = np.array([w / 200., h / 200.], dtype=np.float32)

    scale = scale * 1.25

    return center, scale


def inference_pose_model(model, img, bbox):
    """Inference image(s) with the pose model.

    num_keypoints: K

    Args:
        model (nn.Module): The loaded pose model.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.
        bbox (list|ndarray): Bounding boxes (with scores),
            shaped (4, ) or (5, ). (left, top, width, height, [score])

    Returns:
        predicted poses (ndarray[Kx3]): x, y, score
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = Compose(cfg.valid_pipeline)

    center, scale = box2cs(cfg, bbox)
    # prepare data
    data = {
        'image_file': img,
        'center': center,
        'scale': scale,
        'bbox_score': bbox[4] if len(bbox) == 5 else 1,
        'dataset': 'coco',
        'rotation': 0,
        'imgnum': 0,
        'joints_3d': np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float),
        'joints_3d_visible': np.zeros((cfg.data_cfg.num_joints, 3),
                                      dtype=np.float),
        'ann_info': {
            'image_size':
            cfg.data_cfg['image_size'],
            'num_joints':
            cfg.data_cfg['num_joints'],
            'flip_pairs': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                           [13, 14], [15, 16]],
        }
    }
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        all_preds, _, _ = model(return_loss=False, rescale=True, **data)

    return all_preds[0]


def show_result_pyplot(model,
                       img,
                       result,
                       kpt_score_thr=0.3,
                       skeleton=None,
                       fig_size=(15, 10)):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
        kpt_score_thr (float): The threshold to visualize the keypoints.
        skeleton (list[tuple()]):
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module

    img = model.show_result(
        img, result, skeleton, kpt_score_thr=kpt_score_thr, show=False)

    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.show()


def save_result_visualization(model,
                              img,
                              result,
                              out_file=None,
                              kpt_score_thr=0.3,
                              skeleton=None):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (Tensor or tuple): The results to draw over `img`
                (bbox_result, pose_result).
        kpt_score_thr (float): The threshold to visualize the keypoints.
        skeleton (list[tuple()]):
        out_file (str or None): The filename to write the image.
                Default: None.
    """
    if hasattr(model, 'module'):
        model = model.module

    assert out_file is not None
    model.show_result(
        img,
        result,
        skeleton,
        kpt_score_thr=kpt_score_thr,
        show=False,
        out_file=out_file)
    return 0
