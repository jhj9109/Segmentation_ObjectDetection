from mmdet.apis import set_random_seed

def set_num_classes(cfg, num_classes):
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
                                 with_step=with_step
                                )
                           ]
def set_train(cfg, cfg_name, samples_per_gpu=16, workers_per_gpu=4):

    set_wandb(cfg, 'p32', cfg_name)

    set_save_best_score(cfg)
    
    cfg.data.samples_per_gpu=samples_per_gpu
    cfg.data.workers_per_gpu=workers_per_gpu
    
    cfg.seed = 2020
    set_random_seed(2020, True)

    cfg.gpu_ids = [0]
    cfg.work_dir = f'./work_dirs/{cfg_name}'
    
def set_test(cfg, cfg_name, samples_per_gpu=1, workers_per_gpu=4):

    cfg.data.samples_per_gpu=1
    cfg.data.workers_per_gpu=4
    
    cfg.seed = 2020
    set_random_seed(2020, True)

    cfg.gpu_ids = [0]
    cfg.work_dir = f'./work_dirs/{cfg_name}'