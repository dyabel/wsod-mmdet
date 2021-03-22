from abc import ABCMeta,abstractmethod
import torch.nn as nn
from ..builder import HEADS, build_head, build_roi_extractor
from functools import partial
import torch
from torchvision.models import resnet
from tqdm import tqdm
import logging
import os
from torchvision import transforms, datasets
from PIL import ImageFilter
import random
import torch
from pytorch_metric_learning.utils import logging_presets
from pytorch_metric_learning import losses, miners
import record_keeper
device = torch.device("cuda")

@HEADS.register_module()
class BaseEncoderHead(nn.Module, metaclass=ABCMeta):
    "Base class for Encoder_Head"
    def __init__(self, feature_dim=128, arch='resnet18', bn_splits=8):
        super(BaseEncoderHead, self).__init__()

        # use split batchnorm
        norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else torch.nn.BatchNorm2d
        resnet_arch = getattr(resnet, arch)
        net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)

        self.net = []
        for name, module in net.named_children():
            if name == 'conv1':
                module = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, torch.nn.MaxPool2d):
                continue
            if isinstance(module, torch.nn.Linear):
                self.net.append(torch.nn.Flatten(1))
            self.net.append(module)

        self.net = torch.nn.Sequential(*self.net)


    def forward(self, x):
        x = self.net(x)
        # note: not normalized here
        return x


######################
### from MoCo repo ###
######################
# SplitBatchNorm: simulate multi-gpu behavior of BatchNorm in one gpu by splitting alone the batch dimension
# implementation adapted from https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py
class SplitBatchNorm(torch.nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = torch.nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return torch.nn.functional.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)
