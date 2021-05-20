#!/bin/sh
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=5,6,7 PORT=29502 ./tools/dist_train.sh configs/wsod/wsod_moco_r50_fpn_1x_voc.py 3 --work-dir ../work_dirs/wsod_moco_3fc --name test
