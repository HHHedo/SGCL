import torch
from PIL import Image
from .registry import DATASETS
from .base import BaseDataset


@DATASETS.register_module
class ContrastiveSpatialDataset(BaseDataset):
    """Dataset for contrastive learning methods that forward
        two views of the image at a time (MoCo, SimCLR).
    """

    def __init__(self, data_source, pipeline):
        super(ContrastiveSpatialDataset, self).__init__(data_source, pipeline)

    def __getitem__(self, idx):
        img, idx, bag_idx, x_coord, y_coord = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))
        img1 = self.pipeline(img)
        img2 = self.pipeline(img)
        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
        return dict(img=img_cat,idx=idx,bag_idx=bag_idx, x_coord=x_coord, y_coord=y_coord)

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplemented
