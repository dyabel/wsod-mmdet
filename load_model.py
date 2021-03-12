# -*- coding: utf-8 -*-
# @Time    : 2021/2/26 14:45
# @Author  : duyu
# @Email   : abelazady@foxmail.com
# @File    : load_model.py
# @Software: PyCharm
import torch
import pickle
model = torch.load('data/coco/proposals/rpn_r50_fpn_1x_train2017.pth')
print(model['state_dict'].keys())
# torch.save(model['state_dict'],'data/coco/proposals/rpn_r50_fpn_1x_train2017.pkl')
pickle.dump(model['state_dict'],open('data/coco/proposals/rpn_r50_fpn_1x_train2017.pkl','wb'))
