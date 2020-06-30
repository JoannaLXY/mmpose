import warnings
from abc import ABCMeta, abstractmethod

import cv2
import mmcv
import numpy as np
import torch.nn as nn
from mmcv.image import imwrite
from mmcv.visualization.image import imshow


class BasePose(nn.Module):
    """Base class for pose detectors.

    All recognizers should subclass it.
    All subclass should overwrite:
        Methods:`forward_train`, supporting to forward when training.
        Methods:`forward_test`, supporting to forward when testing.

    Args:
        backbone (dict): Backbone modules to extract feature.
        head (dict): Head modules to give output.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BasePose, self).__init__()

    @abstractmethod
    def forward_train(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def forward_test(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def simple_test(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        pass

    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    pose_kpt_color='red',
                    pose_limb_color=None,
                    radius=4,
                    text_color='green',
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (str or tuple or :obj:`Color`): Color of keypoints.
            pose_limb_color (str or tuple or :obj:`Color`): Color of limbs.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        img_h, img_w, _ = img.shape()

        bbox_result = []
        pose_result = []
        for res in result:
            bbox_result.append(res['bbox'])
            pose_result.append(res['keypoints'])

        bboxes = np.vstack(bbox_result)

        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        # draw bounding boxes
        mmcv.imshow_bboxes(
            img,
            bboxes,
            colors=bbox_color,
            top_k=-1,
            thickness=thickness,
            show=False,
            win_name=win_name,
            wait_time=wait_time,
            out_file=None)

        for person_id, kpts in enumerate(pose_result):
            # draw each point on image
            for kid, kpt in kpts:
                x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]
                if kpt_score > kpt_score_thr:
                    cv2.circle(img, (x_coord, y_coord), radius, pose_kpt_color,
                               thickness)
                    cv2.putText(img, f'{id}', (x_coord, y_coord - 2),
                                cv2.FONT_HERSHEY_COMPLEX, font_scale,
                                text_color)

                # draw limbs
                if skeleton is not None:
                    for sk in skeleton:
                        pos1 = (int(kpt[sk[0] - 1, 0]), int(kpt[sk[0] - 1, 1]))
                        pos2 = (int(kpt[sk[1] - 1, 0]), int(kpt[sk[1] - 1, 1]))
                        if pos1[0] > 0 and pos1[0] < img_w \
                                and pos1[1] > 0 and pos1[1] < img_h \
                                and pos2[0] > 0 and pos2[0] < img_w \
                                and pos2[1] > 0 and pos2[1] < img_h \
                                and kpt[sk[0] - 1, 2] > kpt_score_thr \
                                and kpt[sk[1] - 1, 2] > kpt_score_thr:
                            cv2.line(img, pos1, pos2, pose_kpt_color, 2, 8)

        if show:
            imshow(img, win_name, wait_time)
        if out_file is not None:
            imwrite(img, out_file)
        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img
