import mmcv
proposals = mmcv.load('../../../data/VOCdevkit/proposals/rpn_r101_fpn_voc_train.pkl')
#proposals = mmcv.load('/home/dy20/voc_2007_test.pkl')
# proposals = mmcv.load('test.pkl')
# mmcv.dump(proposals,'test.pkl')
print(len(proposals[0]))
