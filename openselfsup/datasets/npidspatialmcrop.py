import torch
from PIL import Image
from .registry import DATASETS
from .base import BaseDataset
from torchvision.transforms import Compose
from openselfsup.utils import print_log, build_from_cfg
from .registry import DATASETS, PIPELINES
from torch.utils.data import Dataset
from abc import ABCMeta
from .builder import build_datasource
@DATASETS.register_module
class NPIDSpatialMcropDataset(Dataset, metaclass=ABCMeta):
    """Dataset for contrastive learning methods that forward
        two views of the image at a time (MoCo, SimCLR).
    """

    def __init__(self, data_source, pipeline, num_views):
        assert len(num_views) == len(pipeline)
        self.data_source = build_datasource(data_source)
        self.pipeline = []
        for pipe in pipeline:
            tmp_pipeline = Compose([build_from_cfg(p, PIPELINES) for p in pipe])
            self.pipeline.append(tmp_pipeline)
        trans = []
        assert isinstance(num_views, list)
        for i in range(len(num_views)):
            trans.extend([self.pipeline[i]] * num_views[i])
        self.trans = trans

    def __getitem__(self, idx):
        img, idx, bag_idx, x_coord, y_coord = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))
        multi_views = list(map(lambda trans: trans(img), self.trans))
        repeat_idx = [idx for _ in multi_views]
        return dict(img=multi_views[0],img2=multi_views,repeat_idx=repeat_idx, idx=idx,bag_idx=bag_idx, x_coord=x_coord, y_coord=y_coord)

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplemented

    def __len__(self):
        return self.data_source.get_length()

    # def __init__(self, data_source, num_views, pipelines, prefetch=False):
    #     assert len(num_views) == len(pipelines)
    #     self.data_source = build_datasource(data_source)
    #     self.pipelines = []
    #     for pipe in pipelines:
    #         pipeline = Compose([build_from_cfg(p, PIPELINES) for p in pipe])
    #         self.pipelines.append(pipeline)
    #     self.prefetch = prefetch

    #     trans = []
    #     assert isinstance(num_views, list)
    #     for i in range(len(num_views)):
    #         trans.extend([self.pipelines[i]] * num_views[i])
    #     self.trans = trans

import torchvision.transforms as T
@DATASETS.register_module
class NPIDConMcropDataset(Dataset, metaclass=ABCMeta):
    """Dataset for contrastive learning methods that forward
        two views of the image at a time (MoCo, SimCLR).
    """

    def __init__(self, data_source, pipeline, num_views):
        assert len(num_views) == len(pipeline)
        self.data_source = build_datasource(data_source)
        self.pipeline = []
        for pipe in pipeline:
            tmp_pipeline = Compose([build_from_cfg(p, PIPELINES) for p in pipe])
            self.pipeline.append(tmp_pipeline)
        trans = []
        assert isinstance(num_views, list)
        for i in range(len(num_views)):
            trans.extend([self.pipeline[i]] * num_views[i])
        self.trans = trans
        self.f5 = T.FiveCrop(size=(128, 128))

    def __getitem__(self, idx):
        img, idx, bag_idx, x_coord, y_coord = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))
        (top_left, top_right, bottom_left, bottom_right, _) = self.f5(img)
        imgs = [img, top_left, top_right, bottom_left, bottom_right]
        multi_views = list(map(lambda trans, img: trans(img), self.trans, imgs))
        repeat_idx = [idx for _ in multi_views]
        return dict(img=multi_views[0],img2=multi_views,repeat_idx=repeat_idx, idx=idx,bag_idx=bag_idx, x_coord=x_coord, y_coord=y_coord)

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplemented

    def __len__(self):
        return self.data_source.get_length()


@DATASETS.register_module
class NPIDCencropDataset(Dataset, metaclass=ABCMeta):
    """Dataset for contrastive learning methods that forward
        two views of the image at a time (MoCo, SimCLR).
    """

    def __init__(self, data_source, pipeline, num_views):
        assert len(num_views) == len(pipeline)
        self.data_source = build_datasource(data_source)
        self.pipeline = []
        for pipe in pipeline:
            tmp_pipeline = Compose([build_from_cfg(p, PIPELINES) for p in pipe])
            self.pipeline.append(tmp_pipeline)
        trans = []
        assert isinstance(num_views, list)
        for i in range(len(num_views)):
            trans.extend([self.pipeline[i]] * num_views[i])
        self.trans = trans
        self.f1 = T.CenterCrop(size=(128, 128))

    def __getitem__(self, idx):
        img, idx, bag_idx, x_coord, y_coord = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))
        center = self.f1(img)
        imgs = [img, center]
        multi_views = list(map(lambda trans, img: trans(img), self.trans, imgs))
        repeat_idx = [idx for _ in multi_views]
        return dict(img=multi_views[0],img2=multi_views,repeat_idx=repeat_idx, idx=idx,bag_idx=bag_idx, x_coord=x_coord, y_coord=y_coord)

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplemented

    def __len__(self):
        return self.data_source.get_length()


import random
@DATASETS.register_module
class JigsawDataset(Dataset, metaclass=ABCMeta):
    """Dataset for Jigsaw/PIRL
    """

    def __init__(self, data_source, pipeline, num_views):
        assert len(num_views) == len(pipeline)
        self.data_source = build_datasource(data_source)
        self.pipeline = []
        for pipe in pipeline:
            tmp_pipeline = Compose([build_from_cfg(p, PIPELINES) for p in pipe])
            self.pipeline.append(tmp_pipeline)
        trans = []
        assert isinstance(num_views, list)
        for i in range(len(num_views)):
            trans.extend([self.pipeline[i]] * num_views[i])
        self.trans = trans

    def __getitem__(self, idx):
        img, idx, bag_idx, x_coord, y_coord = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))
        sample = T.RandomCrop((255, 255))(img)
        crop_areas = [(i*85, j*85, (i+1)*85, (j+1)*85) for i in range(3) for j in range(3)]
        samples = [sample.crop(crop_area) for crop_area in crop_areas]
        random.shuffle(samples)
        imgs = [img] + samples
        multi_views = list(map(lambda trans, img: trans(img), self.trans, imgs))
        repeat_idx = [idx for _ in multi_views]
        return dict(img=multi_views[0],img2=multi_views,repeat_idx=repeat_idx, idx=idx,bag_idx=bag_idx, x_coord=x_coord, y_coord=y_coord)

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplemented

    def __len__(self):
        return self.data_source.get_length()