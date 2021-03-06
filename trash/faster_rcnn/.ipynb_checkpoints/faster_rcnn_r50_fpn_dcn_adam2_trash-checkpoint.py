_base_ = [
    './faster_rcnn_r50_fpn_adam2_trash.py',
]
model = dict(
    backbone=dict(
        dcn=dict(type='DCN',deform_groups=1,fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))