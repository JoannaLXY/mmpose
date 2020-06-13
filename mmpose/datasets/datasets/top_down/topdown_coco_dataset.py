import logging
import os
import random
import sys

import json_tricks as json
import numpy as np
from pycocotools.coco import COCO

from mmpose.datasets.builder import DATASETS
from ..utils import NullWriter
from .topdown_base_dataset import TopDownBaseDataset

logger = logging.getLogger(__name__)


@DATASETS.register_module
class TopDownCocoDataset(TopDownBaseDataset):
    """CocoDataset dataset for top-down pose estimation.

        The dataset loads raw features and apply specified transforms
        to return a dict containing the image tensors and other information.

        Args:
            ann_file (str): Path to the annotation file.
            img_prefix (str): Path to a directory where images are held.
                Default: None.
            data_cfg (dict): config
            pipeline (list[dict | callable]): A sequence of data transforms.
            test_mode (bool): Store True when building test or
                validation dataset. Default: False.
        """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 test_mode=False):
        super().__init__(
            ann_file, img_prefix, data_cfg, pipeline, test_mode=test_mode)

        self.soft_nms = data_cfg['soft_nms']
        self.nms_thre = data_cfg['nms_thre']
        self.oks_thre = data_cfg['oks_thre']
        self.in_vis_thre = data_cfg['in_vis_thre']
        self.bbox_thre = data_cfg['bbox_thre']

        self.ann_info['flip_pairs'] = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                                       [11, 12], [13, 14], [15, 16]]

        self.ann_info['upper_body_ids'] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.ann_info['lower_body_ids'] = (11, 12, 13, 14, 15, 16)

        self.ann_info['use_different_joints_weight'] = False
        self.ann_info['joints_weight'] = np.array(
            [
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2,
                1.2, 1.5, 1.5
            ],
            dtype=np.float32).reshape((self.ann_info['num_joints'], 1))

        nullwrite = NullWriter()
        oldstdout = sys.stdout
        sys.stdout = nullwrite
        self.coco = COCO(ann_file)
        sys.stdout = oldstdout

        cats = [
            cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())
        ]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict([(self._class_to_coco_ind[cls],
                                             self._class_to_ind[cls])
                                            for cls in self.classes[1:]])
        self.image_set_index = self.coco.getImgIds()
        self.num_images = len(self.image_set_index)
        self.db = self._get_db()

        print('=> num_images: {}'.format(self.num_images))
        print('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        if (not self.test_mode) or self.use_gt_bbox:
            # use ground truth bbox
            gt_db = self._load_coco_keypoint_annotations()
        else:
            # use bbox from detection
            gt_db = self._load_coco_person_detection_results()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        bbox:[x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']
        num_joints = self.ann_info['num_joints']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:

            if max(obj['keypoints']) == 0:
                continue

            joints_3d = np.zeros((num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((num_joints, 3), dtype=np.float)
            for ipt in range(num_joints):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0
            center, scale = self._box2cs(obj['clean_bbox'][:4])
            rec.append({
                'image_file': self.image_path_from_index(index),
                'center': center,
                'scale': scale,
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'dataset': 'coco',
            })

        return rec

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        aspect_ratio = self.ann_info['image_size'][0] / self.ann_info[
            'image_size'][1]
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        dice1 = random.random()
        if (not self.test_mode) and dice1 > 0.7:
            dice_x = random.random()
            center[0] = center[0] + (dice_x - 0.5) * w * 0.4
            dice_y = random.random()
            center[1] = center[1] + (dice_y - 0.5) * h * 0.4

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)

        scale = scale * 1.25

        return center, scale

    def image_path_from_index(self, index):
        """ example: images / train2017 / 000000119993.jpg """
        image_path = os.path.join(self.img_prefix, '%012d.jpg' % index)
        return image_path

    def _load_coco_person_detection_results(self):
        num_joints = self.ann_info['num_joints']
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            logger.error('=> Load %s fail!' % self.bbox_file)
            return None

        print('=> Total boxes: {}'.format(len(all_boxes)))

        kpt_db = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue

            img_name = self.image_path_from_index(det_res['image_id'])
            box = det_res['bbox']
            score = det_res['score']

            if score < self.image_thre:
                continue

            num_boxes = num_boxes + 1

            center, scale = self._box2cs(box)
            joints_3d = np.zeros((num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones((num_joints, 3), dtype=np.float)
            kpt_db.append({
                'image': img_name,
                'center': center,
                'scale': scale,
                'score': score,
                'dataset': 'coco',
                'rotation': 0,
                'imgnum': 0,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
            })
        print('=> Total boxes after fliter low score@{}: {}'.format(
            self.image_thre, num_boxes))
        return kpt_db
