_base_ = [
    './faster_rcnn_r50_fpn_1x_trash.py',
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=11
        )
    )
)
log_config = dict(interval = 49)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

checkpoint_config = dict(max_keep_ckpts=1, interval=1)
