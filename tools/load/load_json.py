import json
import os
# js1 = json.load(open('../data/VOCdevkit/VOC2007/test.json','r'))
# js2 = json.load(open('../data/VOCdevkit/VOC2012/val.json','r'))
#js1 = json.load(open('../data/VOCdevkit/VOC2007/trainval.json','r'))
#js2 = json.load(open('../data/VOCdevkit/VOC2012/train.json','r'))
# js = {}
# for key in js1.keys():
# #    print(type(js1[key]))
#     if isinstance(js1[key],str):
#         js[key] = js1[key]
#     else:
#         js[key] = js1[key]
#         js[key].extend(js2[key])
path = '../data/VOCdevkit/train.json'
# path = '../data/coco/annotations/instances_train2017.json'
js = json.load(open(path,'r'))
print(len(js['images']))
imgs = {}
for img in js['images']:
    imgs[img['id']] = img
print(len(imgs))
# print(js['images'][0].keys())
#print(js['annotations'])
#path = '../data/VOCdevkit/train.json'
# if os.path.exists(path):
#     os.system('rm ' + path)
# json.dump(js,open(path,'w'))

