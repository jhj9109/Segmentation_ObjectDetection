_base_ = [
    './faster_rcnn_r50_fpn_adam2_trash.py',
]
model = dict(
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='soft_nms', iou_threshold=0.5),
            max_per_img=100)))