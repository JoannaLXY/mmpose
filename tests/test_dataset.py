from unittest.mock import MagicMock

import pytest

from mmpose.datasets.builder import DATASETS


@pytest.mark.parametrize('dataset', ['TopDownCocoDataset'])
def test_custom_classes_override_default(dataset):
    dataset_class = DATASETS.get(dataset)
    dataset_class.load_annotations = MagicMock()
    if dataset in ['TopDownCocoDataset']:
        dataset_class.coco = MagicMock()

    # Test setting classes as a tuple
    custom_dataset = dataset_class(
        ann_file=MagicMock(),
        img_prefix='',
        data_cfg=None,
        pipeline=[],
        test_mode=True)

    assert custom_dataset.test_mode is True
