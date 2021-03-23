import mmcv
proposals = mmcv.load('../../../data/VOCdevkit/proposals/rpn_r101_fpn_voc_train_2012.pkl')
# proposals = mmcv.load('test.pkl')
# mmcv.dump(proposals,'test.pkl')
print(type(proposals[0][0][0]))