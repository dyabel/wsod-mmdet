# -*- coding: utf-8 -*-
# @Time    : 2021/3/1 17:48
# @Author  : duyu
# @Email   : abelazady@foxmail.com
# @File    : my_runner.py
# @Software: PyCharm
import time
from mmcv.runner import EpochBasedRunner
import mmcv
import warnings
from mmcv.runner.utils import get_host_info
import torch
import torch.distributed as dist
import multiprocessing
class MyRunner(EpochBasedRunner):
    def __init__(self,*args,**kwargs):
        super(MyRunner,self).__init__(*args,**kwargs)
        self.rank_finish = 0
        # share_value = multiprocessing.Manager().list()
    def train(self,data_loader,**kwargs):
        # for name,param in self.model.named_parameters():
            # print(name)
            # pass
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        # print('dataloader%d'%len(self.data_loader))
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            # if self._inner_iter % 50 ==0:
            #     print('self rank %d iter %d ' % (self.rank,self._inner_iter))
            # if self._inner_iter>531:
            #     print('self rank %d iter %d ' % (self.rank,self._inner_iter))

            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True)
            self.call_hook('after_train_iter')
            self._iter += 1
        print('self rank %d iter %d ' % (self.rank, self._inner_iter))
        # dist.barrier()
        # dist.broadcast(self.rank,self.rank)
        # print('rank_finish')
        # while self.rank_finish < 4:
            # cnt += 1
            # if cnt % 100 == 0:
            #     print(cnt)
            #     print(self.rank_finish)
            # pass
        # time.sleep(10)
        self.call_hook('after_train_epoch')
        self._epoch += 1
    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
           """Start running.

           Args:
               data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                   and validation.
               workflow (list[tuple]): A list of (phase, epochs) to specify the
                   running order and epochs. E.g, [('train', 2), ('val', 1)] means
                   running 2 epochs for training and 1 epoch for validation,
                   iteratively.
           """
           assert isinstance(data_loaders, list)
           assert mmcv.is_list_of(workflow, tuple)
           assert len(data_loaders) == len(workflow)
           if max_epochs is not None:
               warnings.warn(
                   'setting max_epochs in run is deprecated, '
                   'please set max_epochs in runner_config', DeprecationWarning)
               self._max_epochs = max_epochs

           assert self._max_epochs is not None, (
               'max_epochs must be specified during instantiation')

           for i, flow in enumerate(workflow):
               mode, epochs = flow
               if mode == 'train':
                   self._max_iters = self._max_epochs * len(data_loaders[i])
                   break

           work_dir = self.work_dir if self.work_dir is not None else 'NONE'
           self.logger.info('Start running, host: %s, work_dir: %s',
                            get_host_info(), work_dir)
           self.logger.info('workflow: %s, max: %d epochs', workflow,
                            self._max_epochs)
           self.call_hook('before_run')

           while self.epoch < self._max_epochs:
               for i, flow in enumerate(workflow):
                   mode, epochs = flow
                   if isinstance(mode, str):  # self.train()
                       if not hasattr(self, mode):
                           raise ValueError(
                               f'runner has no method named "{mode}" to run an '
                               'epoch')
                       epoch_runner = getattr(self, mode)
                   else:
                       raise TypeError(
                           'mode in workflow must be a str, but got {}'.format(
                               type(mode)))

                   for _ in range(epochs):
                       if mode == 'train' and self.epoch >= self._max_epochs:
                           break
                       epoch_runner(data_loaders[i], **kwargs)
               # time.sleep(4)
               # print('next epoch')

           time.sleep(1)  # wait for some hooks like loggers to finish
           self.call_hook('after_run')