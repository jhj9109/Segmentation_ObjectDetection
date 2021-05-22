optimizer = dict(type='Adam', lr=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[13, 17])
runner = dict(type='EpochBasedRunner', max_epochs=18)