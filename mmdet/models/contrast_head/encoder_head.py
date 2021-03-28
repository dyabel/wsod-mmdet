from abc import ABCMeta,abstractmethod
import torch.nn as nn
from ..builder import HEADS, build_head
from functools import partial
import torch
device = torch.device("cuda")
@HEADS.register_module()
class BaseEncoderHead(nn.Module, metaclass=ABCMeta):
    "Base class for Encoder_Head"
    def __init__(self, feature_dim=128, arch='resnet18', bn_splits=8):
        super(BaseEncoderHead, self).__init__()
        # use split batchnorm
        norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else torch.nn.BatchNorm2d
        self.net = nn.Sequential(
                        # nn.Dropout(),
                        nn.Linear(1024, 512),
                        # nn.MaxPool1d(kernel_size=2, stride=2),
                        nn.ReLU(True),
                        nn.Linear(512, 256),
                        nn.ReLU(True),
                        nn.Linear(256, 128))


    def forward(self, x):
        x = self.net(x)
        x = torch.sigmoid(x)
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
