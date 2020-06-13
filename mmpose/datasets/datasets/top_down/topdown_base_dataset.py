import copy
import logging

import numpy as np
from torch.utils.data import Dataset

from abc import ABCMeta, abstractmethod
from mmpose.datasets.pipelines import Compose

logger = logging.getLogger(__name__)

class TopDownBaseDataset(Dataset, metaclass=ABCMeta):
    """Base class for top-down datasets.

        All top-down datasets should subclass it.
        All subclasses should overwrite:
            Methods:`_get_db`

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

        # set group flag for the sampler
        self.image_info = {}
        self.ann_info = {}

        self.annotations_path = ann_file
        self.img_prefix = img_prefix
        self.pipeline = pipeline
        self.test_mode = test_mode
        self.use_gt_bbox = data_cfg.use_gt_bbox
        self.bbox_file = data_cfg.bbox_file
        self.image_thre = data_cfg.image_thre

        self.ann_info['image_size'] = np.array(data_cfg.image_size)
        self.ann_info['heatmap_size'] = np.array(data_cfg.heatmap_size)
        self.ann_info['num_joints'] = data_cfg.num_joints

        self.ann_info['flip_pairs'] = None

        self.ann_info['model_select_channel'] = data_cfg.model_select_channel
        self.ann_info['num_output_channel'] = data_cfg.num_output_channel
        self.ann_info[
            'model_supervise_channel'] = data_cfg.model_supervise_channel

        self.db = []
        self.pipeline = Compose(self.pipeline)

    @abstractmethod
    def _get_db(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self, ):
        return len(self.db)

    def prepare_train_img(self, idx):
        results = copy.deepcopy(self.db[idx])
        results['ann_info'] = self.ann_info
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        results = copy.deepcopy(self.db[idx])
        results['ann_info'] = self.ann_info
        return self.pipeline(results)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)