_base_ = [
    './schedule_1x.py',
    './sgd.py'
]
lr_config = dict(
    step=[16, 19])
runner = dict(max_epochs=20)
