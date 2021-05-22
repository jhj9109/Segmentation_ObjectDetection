# train.ipynb
- 원본코드를 바탕으로 반복 사용되는 cfg 코드 부분들을 함수화 (set_train)
    - set_options.py에 작성

# inference.ipynb
- 원본코드를 바탕으로 반복 사용되는 cfg 코드 부분들을 함수화 (set_test)
    - set_options.py에 작성
    
# set_options.py
- cfg 코드 부분들을 함수화
    - set_num_classes : 우리 task에 맞게 num_classes를 수정해주는 코드
    - set_save_best_score : 해당 옵션을 켜주는 cfg 설정 코드
    - set_wandb : wandb hook을 설정해주는 코드
    - set_train
    - set_test

# dump_cfg.ipynb & view.ipynb
- 동료 캠퍼에게 제공 받아 활용한 코드
    - dump_cfg.ipynb : 해당 아키텍처의 cfg를 출력하는 mmdet 함수 활용한 코드
    - view.ipynb : inference 결과물인 csv로 bbox 시각화 해주는 코드
    
# set_options2.py
- 학습에 필요한 옵션을 .py 파일이 아닌 cfg에 접근해 수정하는 코드 사용할때 쓰던 코드
- 관리 어려움으로 .py로 cfg 관리하는것으로 변경
    - set_backbone
    - set_neck
    - set_optim_lr
    - set_caffe
    - set_soft_nms

# trash/.py
- cfg를 .py로 관리할때 사용한 파일들
- 아키텍처별 cfg
- dataset.py : base 코드가 우리 task에 맞게 수정된 버전
- dataset_aug1.py : augmentation을 적용하는 pipeline cfg 코드 버전
- dataset_multi_scale_teset.py : (사용하진 못했지만) multi-scale test를 위한 코드