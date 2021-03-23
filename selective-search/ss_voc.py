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
import skimage.io
import selective_search
import json
path_prefix = '../../data/VOCdevkit/VOC2012/'
mode = 'train'
import sys
mode = sys.argv[1]

def work(begin_line,end_line):
    cnt = 0
    inner_cnt = 0
    proposals_list = []
    with open(path_prefix + mode+ '.txt', 'r') as f:
        for line in f.readlines():
            cnt += 1
            if cnt<begin_line:
                continue
            if cnt>end_line:
                break
            inner_cnt += 1
            if inner_cnt % 10 == 0:
                print('process %d gone %d'%(begin_line,inner_cnt))
            img_name = line.strip() + '.jpg'
            image = skimage.io.imread(path_prefix + 'JPEGImages/' + img_name)
            boxes = selective_search.selective_search(image, mode='fast', random_sort=False)
            proposals = []
            # print(len(boxes))
            boxes_filter = selective_search.box_filter(boxes, min_size=20, topN=1000)
            for box in boxes_filter:
                proposal = list(box)
                proposal.append(1)
                proposals.append(proposal)
            proposals_list.append(proposals)

    mmcv.dump(proposals_list,'../../ss_dump_dir12/'+str(begin_line)+mode+'.pkl')

if __name__ == '__main__':
    print(os.cpu_count())
    print(__doc__)
    print(mode)

    model = '../../model.yml.gz'
    # im = cv.imread(sys.argv[2])
    cnt = 0
    process_list = []
    with open(path_prefix + mode+'.txt', 'r') as f:
        total_num = len(f.readlines())
    pro_num = 40
    proposals_list = [[] for i in range(pro_num+1)]
    le = total_num // pro_num
    for n in range(pro_num):
        p = Process(target=work, args=(le*n,le*(n+1)-1))
        process_list.append(p)
        p.start()
    p = Process(target=work, args=(le*pro_num,total_num))
    p.start()
    process_list.append(p)

    for p in process_list:
        p.join()
        print('one process is over +++++')

        # cv.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1, cv.LINE_AA)

    # cv.imshow("edges", edges)
    # cv.imshow("edgeboxes", im)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
