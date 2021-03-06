dataset_type = 'VocMsodDataset'
# data_root = '../data/VOCdevkit/VOC2012/'
data_root = '../data/VOCdevkit/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels','proposals']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img','proposals']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/trainval.json',
        img_prefix=data_root + 'VOC2007/JPEGImages',
        # proposal_file=data_root + 'proposals/ss_trainval_2007.pkl',
        proposal_file=data_root + 'proposals/rpn_r101_fpn_voc_trainval2007.pkl',
        weak_ann_frac=10,
        pipeline=train_pipeline)),
    val=dict(
        type='VocMsodDatasetVal',
        ann_file=data_root + 'VOC2007/test.json',
        img_prefix=data_root + 'VOC2007/JPEGImages',
        proposal_file=data_root + 'proposals/rpn_r101_fpn_voc_trainval2007.pkl',
        # proposal_file=data_root + 'proposals/ss_test_2007.pkl',
        pipeline=test_pipeline),
    test=dict(
        type='VocMsodDatasetVal',
        ann_file=data_root + 'VOC2007/test.json',
        img_prefix=data_root + 'VOC2007/JPEGImages',
        # proposal_file=data_root + 'proposals/ss_test_2007.pkl',
        proposal_file=data_root + 'proposals/rpn_r101_fpn_voc_test2007.pkl',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
