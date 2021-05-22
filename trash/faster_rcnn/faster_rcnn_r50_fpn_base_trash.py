_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../dataset.py',
    '../../_base_/default_runtime.py'
]
# 1. num_classes
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=11
        )
    )
)