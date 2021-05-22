_base_ = './detectors_r50_adam2_trash.py'
model = dict(
    pretrained='open-mmlab://resnext101_32x4d',
    backbone=dict(
        type='DetectoRS_ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        # num_stages=4,
        # out_indices=(0, 1, 2, 3),
        # frozen_stages=1,
        # norm_cfg=dict(type='BN', requires_grad=True),
        # style='pytorch'
        ),
    neck=dict(
        rfp_backbone=dict(
            type='DetectoRS_ResNeXt',
            depth=101,
            groups=32,
            base_width=4,
            # num_stages=4,
            # out_indices=(0, 1, 2, 3),
            # frozen_stages=1,
            # norm_cfg=dict(type='BN', requires_grad=True),
            pretrained='open-mmlab://resnext101_32x4d',
            #style='pytorch'
            ),
        )
    )