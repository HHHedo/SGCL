_base_ = '../../base.py'
# model settings
model = dict(
    type='RelativeLoc',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='RelativeLocNeck',
        in_channels=512,
        out_channels=1024,
        with_avg_pool=True),
    head=dict(
        type='ClsHead',
        with_avg_pool=False,
        in_channels=1024,
        num_classes=8))
# dataset settings
data_source_cfg = dict(
    type='Camelyon',
    memcached=True,
    mclient_path='/mnt/lustre/share/memcached_client')
data_train_list = 'data/camelyon/meta/train_selected.txt'
data_train_root = '/remote-home/share/DATA/CAMELYON16/DATA/train'
dataset_type = 'RelativeLocDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='Resize', size=292),
    dict(type='RandomCrop', size=255),
    dict(type='RandomGrayscale', p=0.66),
]
test_pipeline = [
    dict(type='Resize', size=292),
    dict(type='CenterCrop', size=255),
]
format_pipeline = [
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    imgs_per_gpu=64,  # 64 x 8 = 512
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline,
        format_pipeline=format_pipeline),
    # val=dict(
    #     type=dataset_type,
    #     data_source=dict(
    #         list_file=data_test_list, root=data_test_root, **data_source_cfg),
    #     pipeline=test_pipeline,
    #     format_pipeline=format_pipeline)
        )
# optimizer
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001,
    nesterov=False,
    paramwise_options={
        '\Aneck.': dict(weight_decay=0.0005),
        '\Ahead.': dict(weight_decay=0.0005)})
# learning policy
lr_config = dict(
    policy='step',
    step=[30, 50],
    warmup='linear',
    warmup_iters=5,  # 5 ep
    warmup_ratio=0.1,
    warmup_by_epoch=True)
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 70
