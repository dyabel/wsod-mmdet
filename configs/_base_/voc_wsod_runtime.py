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
# resume_from = 'work_dirs/wsod_voc_with_bba_r50/latest.pth'
resume_from = None
workflow = [('train', 1)]
custom_hooks = [dict(type='MyHook')]
