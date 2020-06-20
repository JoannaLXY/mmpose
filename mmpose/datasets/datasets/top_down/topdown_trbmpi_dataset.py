import json
import os
import random
from collections import OrderedDict, defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmpose.datasets.builder import DATASETS
from .topdown_base_dataset import TopDownBaseDataset


@DATASETS.register_module
class TopDownTRBMPIDataset(TopDownBaseDataset):
    """MPII-trb Dataset dataset for top-down pose estimation.
    paper ref: Haodong Duan et al. TRB: A Novel Triplet Representation
    for Understanding 2D Human Body (ICCV 2019).
    https://github.com/kennymckormick/Triplet-Representation-of-human-Body

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
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']
        self.bbox_thr = data_cfg['bbox_thr']

        # For MPII-trb dataset, only gt bboxes are used.
        assert self.use_gt_bbox

        self.ann_info['flip_pairs'] = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9],
                                       [10, 11], [14, 15]]
        for i in range(6):
            self.ann_info['flip_pairs'].append([16 + i, 22 + i])
            self.ann_info['flip_pairs'].append([28 + i, 34 + i])

        self.ann_info['upper_body_ids'] = [0, 1, 2, 3, 4, 5, 12, 13]
        self.ann_info['lower_body_ids'] = [6, 7, 8, 9, 10, 11]
        self.ann_info['upper_body_ids'].extend(list(range(14, 28)))
        self.ann_info['lower_body_ids'].extend(list(range(28, 40)))

        self.ann_info['use_different_joints_weight'] = False
        self.ann_info['joints_weight'] = np.ones(
            40, dtype=np.float32).reshape((self.ann_info['num_joints'], 1))

        self.trb = COCO(ann_file)

        cats = [cat['name'] for cat in self.trb.loadCats(self.trb.getCatIds())]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_trb_ind = dict(zip(cats, self.trb.getCatIds()))
        self._trb_ind_to_class_ind = dict([(self._class_to_trb_ind[cls],
                                            self._class_to_ind[cls])
                                           for cls in self.classes[1:]])
        self.image_set_index = self.trb.getImgIds()
        self.num_images = len(self.image_set_index)
        self.db = self._get_db()

        print('=> num_images: {}'.format(self.num_images))
        print('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        # For MPII-trb dataset, only gt bbox is used.
        assert self.use_gt_bbox
        # use ground truth bbox
        gt_db = self._load_trb_keypoint_annotations()
        return gt_db

    def _load_trb_keypoint_annotations(self):
        """Ground truth bbox and keypoints """
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_trb_keypoint_annotation_kernal(index))
        return gt_db

    def _load_trb_keypoint_annotation_kernal(self, index):
        """load annotation from COCOAPI

        Note:
            bbox:[x1, y1, w, h]
        Args:
            index: MPII-trb image id
        Returns:
            db entry
        """
        im_ann = self.trb.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']
        num_joints = self.ann_info['num_joints']

        annIds = self.trb.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.trb.loadAnns(annIds)

        if len(objs) == 0:
            return []

        # sanitize bboxes
        valid_objs = []
        if 'bbox' in objs[0]:
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
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float)
            for ipt in range(num_joints):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_visible[ipt, 0] = t_vis
                joints_3d_visible[ipt, 1] = t_vis
                joints_3d_visible[ipt, 2] = 0

            if 'clean_bbox' in obj:
                center, scale = self._box2cs(obj['clean_bbox'][:4])
            else:
                assert 'center' in obj and 'scale' in obj
                center = np.array(obj['center'], dtype=np.float32)
                # obj['scale'] is a float number
                scale = self.ann_info['image_size'] / obj['scale']

            rec.append({
                'image_file': self._image_path_from_index(index),
                'center': center,
                'scale': scale,
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'dataset': 'trb',
                'bbox_score': 1,
            })

        return rec

    def _box2cs(self, box):
        """Get box center & scale given box (x, y, w, h).

        """
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        """This encodes bbox(x,y,w,w) into (center, scale)

        Args:
            x, y, w, h

        Returns:
            center (np.ndarray[float32](2,)): center of the bbox (x, y).
            scale (np.ndarray[float32](2,)): scale of the bbox w & h.
        """
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

    def _image_path_from_index(self, index):
        """ example: images/000000119993.jpg """
        image_path = os.path.join(self.img_prefix, '%09d.jpg' % index)
        return image_path

    def evaluate(self, outputs, res_folder, metrics='mAP', **kwargs):
        """Evaluate trb keypoint results.

        Note:
            num_keypoints: K

        Args:
            outputs(list(preds, boxes, image_path)): Output results.
                preds(np.ndarray[1,K,3]): The first two dimensions are
                    coordinates, score is the third dimension of the array.
                boxes(np.ndarray[1,6]): [center[0], center[1], scale[0]
                    , scale[1],area, score]
                image_path(list[str]): image file path.
            res_folder(str): Path of directory to save the results.
            metrics(str): Metrics to be performed.
                Defaults: 'mAP'.

        Returns:
            name_value (dict): Evaluation results for evaluation metrics.

        """

        res_file = os.path.join(res_folder, 'result_keypoints.json')
        _kpts = []
        for i in range(len(outputs)):
            preds, boxes, image_path = outputs[i]
            str_image_path = ''.join(image_path)

            _kpts.append({
                'keypoints': preds[0],
                'center': boxes[0][0:2],
                'scale': boxes[0][2:4],
                'area': boxes[0][4],
                'score': boxes[0][5],
                'image': int(str_image_path[-13:-4]),
            })

        # image x person x (keypoints)
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)

        results = []
        for v in kpts.values():
            results.append(v)

        self._write_trb_keypoint_results(results, res_file)

        info_str = self._do_python_keypoint_eval(res_file)
        name_value = OrderedDict(info_str)

        return name_value

    def _write_trb_keypoint_results(self, keypoints, res_file):
        """Write results into a json file.

        """
        data_pack = [{
            'cat_id': self._class_to_trb_ind[cls],
            'cls_ind': cls_ind,
            'cls': cls,
            'ann_type': 'keypoints',
            'keypoints': keypoints
        } for cls_ind, cls in enumerate(self.classes)
                     if not cls == '__background__']

        results = self._trb_keypoint_results_one_category_kernel(data_pack[0])

        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def _trb_keypoint_results_one_category_kernel(self, data_pack):
        """Get MPII-trb keypoint results.

        """
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpts[k]['keypoints'] for k in range(len(img_kpts))])
            key_points = np.zeros(
                (_key_points.shape[0], self.ann_info['num_joints'] * 3),
                dtype=np.float)

            for ipt in range(self.ann_info['num_joints']):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]

            result = [{
                'image_id': img_kpts[k]['image'],
                'category_id': cat_id,
                'keypoints': list(key_points[k]),
                'score': img_kpts[k]['score'],
                'center': list(img_kpts[k]['center']),
                'scale': list(img_kpts[k]['scale'])
            } for k in range(len(img_kpts))]
            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using COCOAPI

        """
        trb_dt = self.trb.loadRes(res_file)
        trb_eval = COCOeval(self.trb, trb_dt, 'keypoints')
        trb_eval.params.useSegm = None
        trb_eval.evaluate()
        trb_eval.accumulate()
        trb_eval.summarize()

        stats_names = [
            'AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, trb_eval.stats[ind]))

        return info_str
