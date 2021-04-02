#!/bin/sh
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/wsod/wsod_faster_r50_fpn_1x_voc.py 4 --work-dir ../work_dirs/wsod_faster_voc_frac10
