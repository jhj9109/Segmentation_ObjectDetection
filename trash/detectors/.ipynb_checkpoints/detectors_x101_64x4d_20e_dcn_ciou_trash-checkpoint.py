_base_ = './detectors_x101_64x4d_20e_dcn_trash.py'
model = dict(
    rpn_head=dict(
        loss_bbox=dict(type='CIoULoss')),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                loss_bbox=dict(type='CIoULoss')),
            dict(
                type='Shared2FCBBoxHead',
                loss_bbox=dict(type='CIoULoss')),
            dict(
                type='Shared2FCBBoxHead',
                loss_bbox=dict(type='CIoULoss'))
        ]))