_base_ = '../../../base.py'
# model settings
model = dict(
    type='Classification',
    pretrained=None,
    with_sobel=False,
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN'),
        frozen_stages=4),
    head=dict(
        type='ClsHead', with_avg_pool=True, in_channels=512,
        num_classes=9))
# dataset settings
data_source_cfg = dict(
    type='NCTCRC',
    imagenet_dir='/remote-home/share/DATA/NCTCRC')
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
    imgs_per_gpu=32,  # total 256
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_source=dict(
            train=True,
            **data_source_cfg),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_source=dict(
            train=False,
            **data_source_cfg),
        pipeline=test_pipeline))
# additional hooks
custom_hooks = [
    dict(
        type='ValidateHook',
        dataset=data['val'],
        initial=False,
        interval=10,
        imgs_per_gpu=128,
        workers_per_gpu=4,
        eval_param=dict(topk=(1, 1)))
]
# optimizer
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.) 
# learning policy
lr_config = dict(policy='step', step=[60,80])
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 100