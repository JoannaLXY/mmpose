from abc import ABCMeta, abstractmethod

import torch.nn as nn


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

    def show_result(self):
        return
