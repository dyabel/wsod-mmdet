#!/bin/sh
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29502 ./tools/dist_train.sh configs/wsod/wsod_np_r50_fpn_1x_voc.py 4 --work-dir ../work_dirs/wsod_np_voc --name test
