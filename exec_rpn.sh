#!/bin/sh
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29502 ./tools/dist_train.sh configs/rpn/rpn_r50_fpn_1x_voc.py 4 --work-dir ../work_dirs/rpn_voc --name test
