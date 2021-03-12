# -*- coding: utf-8 -*-
# @Time    : 2021/2/22 18:54
# @Author  : duyu
# @Email   : abelazady@foxmail.com
# @File    : my_hook.py
# @Software: PyCharm
import torch

from mmcv.runner import HOOKS, Hook
@HOOKS.register_module()
class MyHook(Hook):

    def __init__(self):
        pass

    def before_run(self, runner):
        # print(dir(runner))
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass

def hook_fn_backward(module,grad_input,grad_output):
    # print(module)
    # print(module.weight)
    # print(grad_output)
    pass
