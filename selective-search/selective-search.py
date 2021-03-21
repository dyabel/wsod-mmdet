# -*- coding: utf-8 -*-
# @Time    : 2021/2/25 20:29
# @Author  : duyu
# @Email   : abelazady@foxmail.com
# @File    : selective_search.py
# @Software: PyCharm
import cv2
import selectivesearch
import matplotlib.pyplot as plt
import pprint
import matplotlib.patches as mpatches
import skimage.io
import selective_search


# image1="1.jpg"
# #用cv2读取图片
# img = cv2.imread(image1)
# #白底黑字图 改为黑底白字图
# #img=255-img
# #selectivesearch 调用selectivesearch函数 对图片目标进行搜索
# img_lbl, regions =selectivesearch.selective_search(
# img, scale=500, sigma=0.9, min_size=20)
# print(regions) #{'labels': [0.0], 'rect': (0, 0, 585, 301), 'size': 160699} 第一个为原始图的区
# print (len(regions)) #共搜索到199个区域
# # 接下来我们把窗口和图像打印出来，对它有个直观认识
# fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
# ax.imshow(img)
#
# for reg in regions:
#     x, y, w, h = reg['rect']
#     rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
#     ax.add_patch(rect)
# plt.show()
proposals_list = []
import mmcv
cnt = 0
with open('../data/VOCdevkit/train.txt','r') as f:
    for line in f.readlines():
        cnt += 1
        if cnt%10 == 0:
            print(cnt)
        img_name = line.strip() + '.jpg'
        print(img_name)
        # img = cv2.imread('../data/VOCdevkit/JPEGImages/'+img_name)
        # img_lbl, boxes = selectivesearch.selective_search(
        #     img,scale=500,sigma=0.9, min_size=20)
        image = skimage.io.imread('../data/VOCdevkit/JPEGImages/'+img_name)
        boxes = selective_search.selective_search(image,mode='fast', random_sort=False)
        proposals = []
        # print(len(boxes))
        boxes_filter = selective_search.box_filter(boxes, min_size=20, topN=1000)
        for box in boxes_filter:
            proposal = list(box)
            proposal.append(1)
            proposals.append(proposal)
        proposals_list.append(proposals)
mmcv.dump(proposals_list,'../ss_train.pkl')





