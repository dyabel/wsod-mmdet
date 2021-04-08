from scipy.io import matlab
import pickle
import mmcv
mat = matlab.loadmat('/home/dy20/data/VOCdevkit/selective_search_data/voc_2007_trainval.mat')
print(type(mat['boxes'][0][0][0]))
# print(mat['boxes'].tolist())
proposals = []
# print(len(mat['boxes']))
for box in mat['boxes'][0]:
    boxes = []
    for b in box:
        b = b.astype('float64')
        boxes.append(b.tolist())
    proposals.append(boxes)
# pickle.dump(proposals,open('test.pkl','wb'))
# print(len(proposals))
pickle.dump(proposals,open('/home/dy20/data/VOCdevkit/proposals/ss_trainval_2007.pkl','wb'))
# proposals=mmcv.load('test.pkl')
# print(len(proposals))