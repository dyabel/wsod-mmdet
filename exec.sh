#!/bin/sh
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1,2 PORT=29502 ./tools/dist_train.sh configs/wsod/wsod_faster_r50_fpn_1x_voc.py 2 --work-dir ../work_dirs/wsod_faster_voc_3fc --name test
