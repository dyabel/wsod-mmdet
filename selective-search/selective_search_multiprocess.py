#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import sys
from multiprocessing import Process
import os
import mmcv
import cv2
import selectivesearch
import matplotlib.pyplot as plt
import pprint
import matplotlib.patches as mpatches
import skimage.io
import selective_search

def work(begin_line,end_line):
    cnt = 0
    inner_cnt = 0
    proposals_list = []
    with open('../data/VOCdevkit/train.txt', 'r') as f:
        for line in f.readlines():
            if cnt < begin_line:
                continue
            if cnt > end_line:
                break
            inner_cnt += 1
            if inner_cnt % 10 == 0:
                print('process %d gone %d' % (begin_line, inner_cnt))
            img_name = line.strip() + '.jpg'
            print(img_name)
            # img = cv2.imread('../data/VOCdevkit/JPEGImages/'+img_name)
            # img_lbl, boxes = selectivesearch.selective_search(
            #     img,scale=500,sigma=0.9, min_size=20)
            image = skimage.io.imread('../data/VOCdevkit/JPEGImages/' + img_name)
            boxes = selective_search.selective_search(image, mode='fast', random_sort=False)
            proposals = []
            # print(len(boxes))
            boxes_filter = selective_search.box_filter(boxes, min_size=20, topN=1000)
            for box in boxes_filter:
                proposal = list(box)
                proposal.append(1)
                proposals.append(proposal)
            proposals_list.append(proposals)


    mmcv.dump(proposals_list, '../ss_dump_dir/' + str(begin_line) + '.pkl')
if __name__ == '__main__':
    print(os.cpu_count())
    print(__doc__)
    cnt = 0
    process_list = []
    pro_num = 40
    total_num = 10728
    proposals_list = [[] for i in range(pro_num+1)]
    le = total_num // pro_num
    for n in range(pro_num):
        p = Process(target=work, args=(le*n,le*(n+1)))
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