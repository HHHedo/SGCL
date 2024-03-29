_base_ = '../../../base.py'
# model settings
model = dict(
    type='Classification',
    pretrained=None,
    with_sobel=True,
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=2,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN'),
        frozen_stages=4),
    head=dict(
        type='ClsHead', with_avg_pool=True, in_channels=512,
        num_classes=2))
# dataset settings
data_source_cfg = dict(
    type='Camelyon',
    memcached=True,
    mclient_path='/mnt/lustre/share/memcached_client')
data_train_list = 'data/camelyon/meta/train_linear_cls.txt'
data_train_root = 'data/camelyon/train'
data_test_list = 'data/camelyon/meta/val_linear_cls.txt'
data_test_root = 'data/camelyon/val'
dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    imgs_per_gpu=32,  # total 32*8=256, 8GPU linear cls
    workers_per_gpu=5,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_test_list, root=data_test_root, **data_source_cfg),
        pipeline=test_pipeline))
# additional hooks
custom_hooks = [
    dict(
        type='ValidateHook',
        dataset=data['val'],
        initial=True,
        interval=1,
        imgs_per_gpu=128,
        workers_per_gpu=4,
        eval_param=dict(topk=(1, 1), AUC=True))
]
# optimizer
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.) 
# learning policy
lr_config = dict(policy='step', step=[60,80])
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 100
