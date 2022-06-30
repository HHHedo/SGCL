import os
from PIL import Image
import numpy as np
import torch.utils.data as data
from ..registry import DATASOURCES
DIR = '/remote-home/source/DATA/NCTCRC'
@DATASOURCES.register_module
class NCTCRC(object):
    def __init__(self, train=True, imagenet_dir=DIR, image_transforms=None):
        super().__init__()
        split_dir = 'NCT-CRC-HE-100K' if train else 'CRC-VAL-HE-7K'
        self.train = train
        self.root = os.path.join(imagenet_dir, split_dir)
        self.transform = image_transforms
        self.classes, self.class_to_idx = self._find_classes()
        self.instance = self._make_dataset()

    def _find_classes(self):
        classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx 

    def _make_dataset(self):
        instances = []
        labels = []
        directory = os.path.expanduser(self.root)
        for target_class in sorted(self.class_to_idx.keys()):
            class_index = self.class_to_idx[target_class]
            target_dir = np.sort(os.listdir(os.path.join(directory, target_class)))
            for tumor in target_dir:
                path = os.path.join(directory, target_class, tumor)
                item = path, class_index
                instances.append(item)
                labels.append(class_index)
            # print(len(instances))
        self.labels =labels
        return instances

    def get_sample(self, index):
        path, target = self.instance[index]
        sample = Image.open(path).convert('RGB')
        # if self.transform is not None:
        #     sample = self.transform(sample)
        # image_data = [sample, target]
        # # important to return the index!
        # data = image_data + [index]
        # return tuple(data)
        return sample, target

    def get_length(self):
        return len(self.instance)