from __future__ import division
import math

import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler

class WsodDistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # indices = self.dataset.indices
        # for i in self.dataset.cat_strong_ids.keys():
        #     num_strong = len(self.dataset.cat_strong_ids[i])
        #     if num_strong == 0:
        #         continue
        #     for j in range(len(self.dataset.cat_weak_ids[i])):
        #         indices.append([self.dataset.id_idx[self.dataset.cat_strong_ids[i][j%num_strong]],self.dataset.id_idx[self.dataset.cat_weak_ids[i][j]]])
        # indices = np.concatenate(indices)
        # indices = indices.astype(np.int64).tolist()
        # # self.num_samples = len(indices)//2

        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                indice = indice.tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                # pad indice
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[:extra % size])
                indices.extend(indice)

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # if self.num_samples&1 != 0:
        #     self.num_samples += 1
        offset = self.num_samples * self.rank
        if offset > 0:
            indices = indices[offset:offset+self.num_samples]
        else:
            indices = indices[offset:offset + self.num_samples]
        assert self.num_samples == len(indices)
        return iter(indices)




    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch