#!/bin/sh
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29502 ./tools/dist_train.sh configs/wsod/wsod_faster_r50_fpn_1x_voc.py 8 --work-dir ../work_dirs/wsod_moco_voc_frac_3 --name test
