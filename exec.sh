#!/bin/sh
CUDA_LAUNCH_BLOCKING=1  ./tools/dist_train.sh configs/wsod/wsod_faster_r50_fpn_1x_voc.py 1 --work-dir ../work_dirs/wsod_faster_voc_frac_1 --name debug
