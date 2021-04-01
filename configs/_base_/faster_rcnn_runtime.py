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
# resume_from = 'work_dirs/faster_rcnn_r50_fpn_1x_coco/epoch_4.pth'
# resume_from = '../work_dirs/faster_rcnn_voc/latest.pth'
resume_from = None
workflow = [('train', 1)]
