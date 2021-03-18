# -*- coding: utf-8 -*-
# @Time    : 2021/3/1 17:48
# @Author  : duyu
# @Email   : abelazady@foxmail.com
# @File    : my_runner.py
# @Software: PyCharm
import time
from mmcv.runner import EpochBasedRunner
class MyRunner(EpochBasedRunner):
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
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True)
            self.call_hook('after_train_iter')
            self._iter += 1
        # print('self rank %d ' % self.rank)
        # time.sleep(8)
        self.call_hook('after_train_epoch')
        self._epoch += 1
