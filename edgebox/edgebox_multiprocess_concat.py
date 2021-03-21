#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This sample demonstrates structured edge detection and edgeboxes.
Usage:
  edgeboxes_demo.py [<model>] [<input_image>]
'''

import cv2 as cv
import numpy as np
import sys
from multiprocessing import Process
import os
import mmcv



if __name__ == '__main__':
    print(os.cpu_count())
    print(__doc__)
    path_prefix = '../../data/VOCdevkit/'
    with open(path_prefix + 'train.txt', 'r') as f:
        total_num = len(f.readlines())

    model = '../../model.yml.gz'
    # im = cv.imread(sys.argv[2])
    cnt = 0
    process_list = []
    pro_num = 40
    le = total_num // pro_num
    proposals_list_concat = []
    for i in range(pro_num+1):
        proposals_list = mmcv.load('../../edgebox_dump_dir/' + str(i*le) + '.pkl')
        proposals_list_concat.extend(proposals_list)
    print(len(proposals_list_concat))
    mmcv.dump(proposals_list_concat,'../../data/VOCdevkit/proposals/edgebox_voc_train.pkl')





        # cv.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1, cv.LINE_AA)

    # cv.imshow("edges", edges)
    # cv.imshow("edgeboxes", im)
    # cv.waitKey(0)
    # cv.destroyAllWindows()