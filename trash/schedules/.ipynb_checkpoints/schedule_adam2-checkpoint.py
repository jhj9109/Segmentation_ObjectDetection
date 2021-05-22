_base_ = [
    './schedule_1x.py',
]
optimizer = dict(type='Adam', lr=0.0001)

lr_config = dict(step=[13, 16])
runner = dict(max_epochs=18)