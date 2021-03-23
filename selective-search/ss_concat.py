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
mode = 'train'

import sys
mode = sys.argv[1]

if __name__ == '__main__':
    print(mode)
    print(os.cpu_count())
    print(__doc__)
    path_prefix = '../../data/VOCdevkit/VOC2012/'
    with open(path_prefix + mode+'.txt', 'r') as f:
        total_num = len(f.readlines())


    # im = cv.imread(sys.argv[2])
    cnt = 0
    process_list = []
    pro_num = 40
    le = total_num // pro_num
    proposals_list_concat_raw = []
    proposals_list_concat = []
    for i in range(pro_num+1):
        proposals_list = mmcv.load('../../ss_dump_dir12/' + str(i*le) + mode+'.pkl')
        proposals_list_concat_raw.extend(proposals_list)
    for proposals_list in proposals_list_concat_raw:
        proposals_list_concat.append(np.array(proposals_list,dtype=np.float32))
    print(proposals_list_concat[0].shape)

    mmcv.dump(proposals_list_concat,'../../data/VOCdevkit/proposals/ss_voc_'+mode+'_12.pkl')





        # cv.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1, cv.LINE_AA)

    # cv.imshow("edges", edges)
    # cv.imshow("edgeboxes", im)
    # cv.waitKey(0)
    # cv.destroyAllWindows()