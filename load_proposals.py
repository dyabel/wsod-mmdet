import mmcv
proposals = mmcv.load('../data/VOCdevkit/proposals/rpn_r101_fpn_voc_train.pkl')
# proposals = mmcv.load('test.pkl')
# mmcv.dump(proposals,'test.pkl')
print(proposals)