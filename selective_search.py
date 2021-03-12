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


image1="1.jpg"
#用cv2读取图片
img = cv2.imread(image1)
#白底黑字图 改为黑底白字图
#img=255-img
#selectivesearch 调用selectivesearch函数 对图片目标进行搜索
img_lbl, regions =selectivesearch.selective_search(
img, scale=500, sigma=0.9, min_size=20)
print(regions) #{'labels': [0.0], 'rect': (0, 0, 585, 301), 'size': 160699} 第一个为原始图的区
print (len(regions)) #共搜索到199个区域
# 接下来我们把窗口和图像打印出来，对它有个直观认识
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(img)

for reg in regions:
    x, y, w, h = reg['rect']
    rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)
plt.show()
