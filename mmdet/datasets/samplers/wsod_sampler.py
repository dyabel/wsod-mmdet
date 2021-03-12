from __future__ import division
import math

import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler


class WsodSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += int(np.ceil(
                size / self.samples_per_gpu)) * self.samples_per_gpu

    def __iter__(self):
        indices = []
        for i in self.dataset.cat_strong_ids.keys():
            num_strong = len(self.dataset.cat_strong_ids[i])
            if num_strong == 0:
                continue
            for j in range(len(self.dataset.cat_weak_ids[i])):
                indices.append([self.dataset.id_idx[self.dataset.cat_strong_ids[i][j%num_strong]],self.dataset.id_idx[self.dataset.cat_weak_ids[i][j]]])
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples


