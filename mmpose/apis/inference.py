import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmpose.core.evaluation import keypoints_from_heatmaps
from mmpose.datasets.pipelines import Compose
from mmpose.models import build_posenet


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
    # load model checkpoint
    load_checkpoint(model, checkpoint)
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

    Args:
        model (nn.Module): The loaded pose model.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.
        bbox (ndarray): Bounding boxes (with scores), shaped (n, 4) or (n, 5).
            (left, top, width, height, [score])

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
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
        'bbox_score': bbox[4],
        'dataset': 'coco',
        'rotation': 0,
        'imgnum': 0,
        'joints_3d': np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float),
        'joints_3d_visible': np.zeros((cfg.data_cfg.num_joints, 3),
                                      dtype=np.float)
    }
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    pose_result = []
    # forward the model
    with torch.no_grad():
        heatmaps = model(return_loss=False, rescale=True, **data)

        coords, scores = keypoints_from_heatmaps(
            heatmaps,
            center,
            scale,
            post_process=True,
            unbiased=False,
            kernel=11)

        pose_result.append(np.concatenate((coords, scores), 2))

    return pose_result


def show_result_pyplot(model,
                       img,
                       result,
                       score_thr=0.3,
                       skeleton=None,
                       fig_size=(15, 10)):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
        score_thr (float): The threshold to visualize the keypoints.
        skeleton (list[tuple()]):
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module

    img = model.show_result(
        img, result, skeleton, score_thr=score_thr, show=False)

    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.show()


def save_result_visualization(model,
                              img,
                              result,
                              out_file=None,
                              score_thr=0.3,
                              skeleton=None):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (Tensor or tuple): The results to draw over `img`
                (bbox_result, pose_result).
        score_thr (float): The threshold to visualize the keypoints.
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
        score_thr=score_thr,
        show=False,
        out_file=out_file)
    return 0
