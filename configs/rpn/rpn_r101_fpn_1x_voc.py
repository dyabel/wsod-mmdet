_base_ = './rpn_r50_fpn_1x_voc.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
