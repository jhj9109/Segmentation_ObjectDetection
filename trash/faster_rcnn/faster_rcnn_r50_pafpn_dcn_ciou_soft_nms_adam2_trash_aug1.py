_base_ = './faster_rcnn_r50_fpn_dcn_ciou_soft_nms_adam2_trash_aug1.py'

model = dict(
    neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5))