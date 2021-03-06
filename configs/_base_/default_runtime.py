checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
# resume_from = '/data/dy/work_dirs/fast_rcnn_coco/latest.pth'
# resume_from = '/data1/dataset/dy/work_dirs/fast_rcnn_voc12/latest.pth'
workflow = [('train', 1)]
