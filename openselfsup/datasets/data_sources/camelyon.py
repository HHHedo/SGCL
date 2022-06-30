import os
from PIL import Image

from ..registry import DATASOURCES
from .utils import McLoader
import pickle

@DATASOURCES.register_module
class Camelyon(object):
    def __init__(self, root, list_file, memcached=False, mclient_path=None):
        with open(list_file, 'r') as f:
            lines = f.readlines()
        self.has_labels = len(lines[0].split()) == 2
        # if self.has_labels:
        #     self.fns, self.labels = zip(*[l.strip().split() for l in lines])
        #     self.labels = [int(l) for l in self.labels]
        # else:
        #     self.fns = [l.strip() for l in lines]

        self.fns, self.bags = zip(*[(l.strip().split()[0],l.strip().split('/')[1]) for l in lines])
        self.bags_to_idx = {bag:idx for idx,bag in enumerate(sorted(set(self.bags)))}
        self.labels = [int(1) if patch.split('/')[-3].startswith('tumor') else int(0) for patch in self.fns]
        self.bag_idxs = [self.bags_to_idx[bag] for bag in self.bags]
        self.x_coords = [int(patch.split('_')[-3]) for patch in self.fns]
        self.y_coords = [int(patch.split('_')[-2]) for patch in self.fns]

        self.fns = [os.path.join(root, fn) for fn in self.fns]
        
        self.memcached = memcached
        self.mclient_path = mclient_path
        self.initialized = False
        # self.redis = redis

    def _init_memcached(self):
        if not self.initialized:
            assert self.mclient_path is not None
            self.mc_loader = McLoader(self.mclient_path)
            self.initialized = True

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        if self.memcached:
            self._init_memcached()
        if self.memcached:
            img = self.mc_loader(self.fns[idx])
        # elif self.redis:
        #     img=pickle.loads(self.redis.get(self.fns[idx]))
        else:
            img = Image.open(self.fns[idx])
        # print(idx, self.fns[idx])
        img = img.convert('RGB')
        bag_idx = self.bag_idxs[idx]
        x_coord = self.x_coords[idx]
        y_coord = self.y_coords[idx]
        if self.has_labels:
            target = self.labels[idx]
            return img, target
        else:
            return img, idx, bag_idx, x_coord, y_coord
