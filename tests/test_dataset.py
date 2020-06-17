from unittest.mock import MagicMock

import pytest

from mmpose.datasets import DATASETS


@pytest.mark.parametrize('dataset', ['TopDownCocoDataset'])
def test_custom_classes_override_default(dataset):
    # test datasets
    dataset_class = DATASETS.get(dataset)
    dataset_class.load_annotations = MagicMock()
    if dataset in ['TopDownCocoDataset']:
        dataset_class.coco = MagicMock()

    channel_cfg = dict(
        num_joints=17,
        sub_data_name=['coco'],
        dataset_channel=[
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        ],
        inference_channel=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        ])

    data_cfg = dict(
        image_size=[192, 256],
        heatmap_size=[48, 64],
        num_output_channels=channel_cfg['num_joints'],
        num_joints=channel_cfg['num_joints'],
        dataset_channel=channel_cfg['dataset_channel'],
        inference_channel=channel_cfg['inference_channel'],
        soft_nms=False,
        nms_thre=1.0,
        oks_thre=0.9,
        in_vis_thre=0.2,
        bbox_thre=1.0,
        use_gt_bbox=True,
        image_thre=0.0,
        bbox_file='',
    )

    # Test setting classes as a tuple
    custom_dataset = dataset_class(
        ann_file='tests/data/test.json',
        img_prefix='',
        data_cfg=data_cfg,
        pipeline=[],
        test_mode=True)

    assert custom_dataset.test_mode is True
