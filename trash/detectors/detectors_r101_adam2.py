_base_ = './detectors_r50_adam2_trash.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))