from mmcv.utils import Registry, build_from_cfg
from torch import nn

BACKBONES = Registry('backbone')
HEADS = Registry('head')
LOSSES = Registry('loss')
POSENETS = Registry('posenet')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_head(cfg):
    return build(cfg, HEADS)


def build_loss(cfg):
    return build(cfg, LOSSES)


def build_posenet(cfg):
    return build(cfg, POSENETS)
