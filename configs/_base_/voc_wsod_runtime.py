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
# resume_from = '/data1/dataset/dy/work_dirs/wsod_faster_voc/latest.pth'
# resume_from = '../work_dirs/wsod_faster_voc_frac_1/epoch_4.pth'
workflow = [('train', 1)]
custom_hooks = [dict(type='MyHook')]
