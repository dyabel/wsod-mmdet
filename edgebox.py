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

if __name__ == '__main__':
    print(__doc__)

    model = '../model.yml.gz'
    # im = cv.imread(sys.argv[2])
    cnt = 0
    proposals_list = []
    with open('../data/VOCdevkit/train.txt', 'r') as f:
        for line in f.readlines():
            cnt += 1
            if cnt % 10 == 0:
                print(cnt)
            img_name = line.strip() + '.jpg'
            im = cv.imread('../data/VOCdevkit/JPEGImages/' + img_name)
            edge_detection = cv.ximgproc.createStructuredEdgeDetection(model)
            rgb_im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
            edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

            orimap = edge_detection.computeOrientation(edges)
            edges = edge_detection.edgesNms(edges, orimap)

            edge_boxes = cv.ximgproc.createEdgeBoxes()
            edge_boxes.setMaxBoxes(30)
            boxes = edge_boxes.getBoundingBoxes(edges, orimap)

            proposals = []
            for i,b in enumerate(boxes[0]):
                x, y, w, h = b
                proposals.append([x,y,w,h,boxes[1][i]])
            proposals_list.append(proposals)
    import mmcv
    mmcv.dump(proposals_list,'../voc_train_edgebox.pkl')
        # cv.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1, cv.LINE_AA)

    # cv.imshow("edges", edges)
    # cv.imshow("edgeboxes", im)
    # cv.waitKey(0)
    # cv.destroyAllWindows()