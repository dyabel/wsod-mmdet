#!/bin/sh
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc.py 4 --work-dir ../work_dirs/faster_rcnn_voc --name debug
