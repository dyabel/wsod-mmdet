_base_ = './rpn_r50_fpn_1x_voc12.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
