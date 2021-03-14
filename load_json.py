import json
js1 = json.load(open('/home/duyu/data/VOCdevkit/VOC2007/test.json','r'))
js2 = json.load(open('/home/duyu/data/VOCdevkit/VOC2012/val.json','r'))
js = {}
for key in js1.keys():
    if isinstance(key,str):
        js[key] = js1[key]
    else:
        js[key] = js1[key].extend(js2[key])
json.dump(js,open('val.json','w'))

