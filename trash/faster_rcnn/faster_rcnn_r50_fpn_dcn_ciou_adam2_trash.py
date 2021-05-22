_base_ = [
    './faster_rcnn_r50_fpn_dcn_adam2_trash.py',
]
model = dict(
    rpn_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(type='CIoULoss', loss_weight=1.0)),
    roi_head=dict(
        bbox_head=dict(reg_decoded_bbox=True,
                       loss_bbox=dict(type='CIoULoss', loss_weight=1.0))
    )
)