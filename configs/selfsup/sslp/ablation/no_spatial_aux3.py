_base_ = '../../../base.py'
# model settings
model = dict(
    type='SpatialCL4',
    pretrained=None,
    loss_lambda=1/2,
    rampup_length=10,
    similar=True,
    no_clusters=1000, 
    no_kmeans=3,
    dis_threshold=3,
    aux_num=3,
    k=65536, #no. of negative samples
    nei_k = int(4096*2/(1+3)), # change with k & aux_num!!!!!!!!!!!!!!!!!!!!!!!!!!!1
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='NonLinearNeckV1',
        in_channels=512,
        hid_channels=512,
        out_channels=128,
        with_avg_pool=False),
    head=dict(type='ContrastiveHead', temperature=0.2), 
    memory_bank=dict(
        type='SimpleMemory', length=681486, feat_dim=128, momentum=0.5),
    memory_bank_b=dict(
        type='SimpleMemory', length=681486, feat_dim=512, momentum=0.5))
# dataset settings
data_source_cfg = dict(
    type='Camelyon',
    memcached=True,
    mclient_path='/mnt/lustre/share/memcached_client')
data_train_list = 'data/camelyon/meta/train_selected.txt'
data_train_root = '/remote-home/share/DATA/CAMELYON16/DATA/train'
# data_train_root = '/root/data/camelyon/train'
dataset_type = 'NPIDSpatialDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='ColorJitter',
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.4),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
# train_pipeline = [
#     dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
#     dict(
#         type='RandomAppliedTrans',
#         transforms=[
#             dict(
#                 type='ColorJitter',
#                 brightness=0.4,
#                 contrast=0.4,
#                 saturation=0.4,
#                 hue=0.1)
#         ],
#         p=0.8),
#     dict(type='RandomGrayscale', p=0.2),
#     dict(
#         type='RandomAppliedTrans',
#         transforms=[
#             dict(
#                 type='GaussianBlur',
#                 sigma_min=0.1,
#                 sigma_max=2.0)
#         ],
#         p=0.5),
#     dict(type='RandomHorizontalFlip'),
#     dict(type='ToTensor'),
#     dict(type='Normalize', **img_norm_cfg),
# ]
test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    imgs_per_gpu=32,  # total 32*8=256
    workers_per_gpu=5,
    # drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.015, weight_decay=0.0001, momentum=0.9)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 100
