_base_ = [
    './default_runtime.py',
]
checkpoint_config = dict(interval=1)
log_config = dict(hooks=[dict(type='TextLoggerHook'),
                         dict(type='WandbLoggerHook',
                              init_kwargs=dict(project='p32'),
                              with_step=False)]
                  interval=49
                 )
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]