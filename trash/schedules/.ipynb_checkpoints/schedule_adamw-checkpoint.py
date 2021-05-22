_base_ = [
    './schedule_1x.py',
]
optimizer = dict(_delete_=True, type='AdamW', lr=0.000025, weight_decay=0.0001)

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1, norm_type=2))

lr_config = dict(step=[8, 11])
runner = dict(max_epochs=12)

