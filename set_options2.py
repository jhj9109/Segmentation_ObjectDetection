from mmdet.apis import set_random_seed

classes = ("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass",
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
PREFIX = '../../input/data/'

# [('bbox_mAP', 0.013), ('bbox_mAP_50', 0.042), ('bbox_mAP_75', 0.004), ('bbox_mAP_s', 0.0), ('bbox_mAP_m', 0.018), ('bbox_mAP_l', 0.025), ('bbox_mAP_copypaste', '0.013 0.042 0.004 0.000 0.018 0.025')])

def set_optim(cfg, mode, options=dict() ):
    if mode == 'adam':
        cfg.optimizer = dict(
            type='Adam',
            lr=0.0001
        )
def set_num_classes(cfg, num_classes=11):
    if 'roi_head' in cfg.model:
        if 'bbox_head' in cfg.model.roi_head and 'num_classes' in cfg.model.roi_head.bbox_head: # for roi (2-stage detector)
            cfg.model.roi_head.bbox_head.num_classes = num_classes 
            print('cfg.model.roi_head.bbox_head.num_classes = 11')
            return 
        if 'semantic_head' in cfg.model.roi_head and 'num_classes' in cfg.model.roi_head.semantic_head: # for htc
            cfg.model.roi_head.semantic_head.num_classes = num_classes
            print ('cfg.model.roi_head.semantic_head.num_classes = 11')
            return
        if 'mask_head' in cfg.model.roi_head and 'num_classes' in cfg.model.roi_head.mask_head: # for mask R-CNN
            cfg.model.roi_head.mask_head.num_classes = num_classes 
            print ('cfg.model.roi_head.mask_head.num_classes = 11')
            return
    if 'bbox_head' in cfg.model and 'num_classes' in cfg.model.bbox_head:
        cfg.model.bbox_head.num_classes = num_classes
        print('cfg.model.bbox_head.num_classes = 11')
        return
    print('Fail')
    return

def set_save_best_score(cfg):
    cfg.evaluation = dict(interval=1, metric="bbox", save_best="bbox_mAP_50", rule="greater")
    
def set_wandb(cfg, project_name, run_name, with_step=False):
    cfg.log_config.hooks = [dict(type='TextLoggerHook'),
                            dict(type='WandbLoggerHook',
                                 init_kwargs=dict(project='p32', name=run_name),
                                 with_step=False
                                )
                           ]
def set_p3_test(cfg):
    
    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = PREFIX
    cfg.data.test.ann_file = PREFIX + 'test.json'
    cfg.data.test.pipeline[1]['img_scale'] = (512, 512)
    
    set_num_classes(cfg, num_classes=11)

    cfg.seed = 2020
    set_random_seed(2020, True)

    cfg.gpu_ids = [0]
    
def set_p3_train(cfg, save_file_name, samples_per_gpu=16, wokers_per_gpu=4, max_keep_ckpts=1):
    set_wandb(cfg, 'p32', save_file_name)
    
    set_save_best_score(cfg)
    
    # trash/dataset.py 에서 전부 진행    
#     cfg.data.train.classes = classes
#     cfg.data.train.img_prefix = PREFIX
#     cfg.data.train.ann_file = PREFIX + 'train.json'
#     cfg.data.train.pipeline[2]['img_scale'] = (512, 512)

#     cfg.data.val.classes = classes
#     cfg.data.val.img_prefix = PREFIX
#     cfg.data.val.ann_file = PREFIX + 'val.json'
#     cfg.data.val.pipeline[1]['img_scale'] = (512, 512)

#     set_num_classes(cfg, num_classes=11)

    cfg.data.samples_per_gpu = samples_per_gpu
    cfg.wokers_per_gpu = wokers_per_gpu

    cfg.seed = 2020
    set_random_seed(2020, True)

    cfg.gpu_ids = [0]
    cfg.work_dir = f'./work_dirs/{save_file_name}'

    cfg.optimizer_config.grad_clip['max_norm']=35
    cfg.optimizer_config.grad_clip['norm_type']=2
    #  cfg.optimizer_config._delete_=True for trash
    cfg.checkpoint_config = dict(max_keep_ckpts=max_keep_ckpts, interval=1)

# 1. backbone
def set_backbone(cfg, options):
    
    def set_x101(cfg, num_groups):
        if num_groups not in [32, 64]:
            print(f'set_x101 Fail at num_groups:{num_groups}')
        cfg.model.backbone.groups = num_groups
        cfg.model.pretrained=f'open-mmlab://resnext101_{num_groups}x4d'
        cfg.model.backbone.type = 'ResNeXt'
        cfg.model.backbone.depth = 101
        cfg.model.backbone.groups = 32
        cfg.model.backbone.base_width = 4
        cfg.model.backbone.num_stages = 4
        cfg.model.backbone.out_indices=(0,1,2,3)
        cfg.model.backbone.frozen_stages=1
        cfg.model.backbone.norm_cfg.type='BN'
        cfg.model.backbone.requires_grad=True
        cfg.model.backbone.style='pytorch'
        
    if 'caffe' in options:
        pass
    elif 'r101' in options:
        cfg.model.pretrained='torchvision://resnet101'
        cfg.model.backbone.depth=101
    elif 'x101_32x4d' in options:
        set_x101(cfg, 32)
    elif 'x101_64_4d' in options:
        set_x101(cfg, 64)
        
# 2. neck
def set_neck(cfg, options):
    if 'bifpn' in options:
        model = dict(neck=dict(type='BiFPN',
                               in_channels=[256, 512, 1024, 2048],
                               out_channels=256,
                               strides=[4, 8, 16, 32],
                               norm_cfg=dict(type='BN', requires_grad=True),
                               num_outs=5))
        print(f'set neck bifpn')
    elif 'pafpn' in options:
        cfg.model.neck.type = 'PAFPN'

# 3. optimizer & lr_config
def set_optim_lr(cfg, options):
    for i in [1,2,3]:
        if f'{i}x' in options:
            cfg.optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
            cfg.lr_config = dict(
                    policy='step',
                    warmup='linear',
                    warmup_iters=500,
                    warmup_ratio=0.001,
                    step=[8*i, 11*i])
            cfg.runner.type='EpochBasedRunner'
            cfg.runner.max_epochs=12*i
            return
    else:
        if '20e' in options:
            cfg.optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
            cfg.optimizer_config = dict(grad_clip=None)
            cfg.lr_config = dict(
                    policy='step',
                    warmup='linear',
                    warmup_iters=500,
                    warmup_ratio=0.001,
                    step=[16, 19])
            cfg.runner.type='EpochBasedRunner'
            cfg.runner.max_epochs=20
        elif 'adam2' in options:
            cfg.optimizer = dict(type='Adam', lr=0.0001)
            cfg.lr_config.step = [13, 17]
            cfg.runner.max_epochs = 20
            
# 4. pytorch(default) or caffe(long train & multiscale train)
def set_caffe(cfg, options):
    if 'caffe' in options:
        caffe_img_norm_cfg = dict(
            mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
        if 'r101' not in options:
            cfg.model.pretrained = 'open-mmlab://detectron2/resnet50_caffe'
        else:
            cfg.model.pretrained = 'open-mmlab://detectron2/resnet101_caffe'
        cfg.model.backbone.norm_cfg.requires_grad=False
        cfg.model.backbone.norm_eval=True
        cfg.model.backbone.style='caffe'

        cfg.data.train.pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **caffe_img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]
        # mstrain
        if 'mstrain' in options:
            cfg.data.train.pipeline[2]['img_scale'] = [
                (512, 412), (512, 432), (512, 452), (512, 472), (512, 492), (512, 512)
            ]
            cfg.data.train.pipeline[2]['multiscale_mode'] = 'value'
        cfg.data.val.pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **caffe_img_norm_cfg),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img']),
                ])
        ]
        cfg.data.test.pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **caffe_img_norm_cfg),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img']),
                ])
        ]
        cfg.img_norm_cfg.mean = caffe_img_norm_cfg['mean']
        cfg.img_norm_cfg.std=caffe_img_norm_cfg['std']
        cfg.img_norm_cfg.to_rgb=caffe_img_norm_cfg['to_rgb']

def set_soft_nms(cfg, options):
    if 'soft_nms' in options:
        cfg.model.test_cfg.rcnn.nms.type='soft_nms'
        cfg.model.test_cfg.rcnn.nms.min_score=0.05

def set_options(cfg, options):
    # 1.Backbone
    set_backbone(cfg, options)
    
    # 2.Neck
    set_neck(cfg, options)
        
    # 3.optimizer & lr_config
    set_optim_lr(cfg, options)
    
    # 4.caffe
    set_caffe(cfg, options)
    
    # 5. soft_nms
    set_soft_nms(cfg, options)