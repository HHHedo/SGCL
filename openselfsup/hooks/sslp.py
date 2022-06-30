import numpy as np

from mmcv.runner import Hook

import torch
import torch.distributed as dist

from openselfsup.third_party import clustering as _clustering
from openselfsup.utils import print_log
from .registry import HOOKS
from .extractor import Extractor


@HOOKS.register_module
class SSLPHOOK(Hook):
    """Hook for DeepCluster.

    Args:
        extractor (dict): Config dict for feature extraction.
        clustering (dict): Config dict that specifies the clustering algorithm.
        unif_sampling (bool): Whether to apply uniform sampling.
        reweight (bool): Whether to apply loss re-weighting.
        reweight_pow (float): The power of re-weighting.
        init_memory (bool): Whether to initialize memory banks for ODC.
            Default: False.
        initial (bool): Whether to call the hook initially. Default: True.
        interval (int): Frequency of epochs to call the hook. Default: 1.
        dist_mode (bool): Use distributed training or not. Default: True.
        data_loaders (DataLoader): A PyTorch dataloader. Default: None.
    """

    def __init__(
            self,
            init_memory=False,  # for ODC
            initial=True,
            interval=1,
            dist_mode=True,
            data_loaders=None):
        self.init_memory = init_memory
        self.initial = initial
        self.interval = interval
        self.dist_mode = dist_mode
        self.data_loaders = data_loaders

    def before_run(self, runner):
        if self.initial:
            self.deepcluster(runner)

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        self.deepcluster(runner)
        
    @torch.no_grad()
    def deepcluster(self, runner):
        # step 1: get features
        print('Fitting K-means with FAISS')
        features = runner.model.module.memory_bank.feature_bank
        kmeans = runner.model.module.kmeans
        # clustering_algo = _clustering.__dict__[self.clustering_type](
        #         **self.clustering_cfg)
        #     # Features are normalized during clustering
        # clustering_algo.cluster(features, verbose=True)
        # assert isinstance(clustering_algo.labels, np.ndarray)
        # cluster_labels = clustering_algo.labels.astype(np.int64)
        if runner.rank == 0:
            km = Kmeans(kmeans, features, [0,1,2,3])
            # Features are normalized during clustering
            cluster_labels = km.compute_clusters()

            # step 3: assign new labels
            runner.model.module._cluster_update(cluster_labels)

    def evaluate(self, runner, new_labels):
        hist = np.bincount(new_labels, minlength=self.clustering_cfg.k)
        empty_cls = (hist == 0).sum()
        minimal_cls_size, maximal_cls_size = hist.min(), hist.max()
        if runner.rank == 0:
            print_log(
                "empty_num: {}\tmin_cluster: {}\tmax_cluster:{}".format(
                    empty_cls.item(), minimal_cls_size.item(),
                    maximal_cls_size.item()),
                logger='root')


import faiss
import torch
import numpy as np

DEFAULT_KMEANS_SEED = 1234

def run_kmeans(x, nmb_clusters, verbose=False,
               seed=DEFAULT_KMEANS_SEED, gpu_device=0):
    """
    Runs kmeans on 1 GPU.
    
    Args:
    -----
    x: data
    nmb_clusters (int): number of clusters
    
    Returns:
    --------
    list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    clus.seed = seed
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = gpu_device

    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    losses = faiss.vector_to_array(clus.obj)
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], losses[-1]


def run_kmeans_multi_gpu(x, nmb_clusters, verbose=False, seed=DEFAULT_KMEANS_SEED, gpu_device=0):

    """
    Runs kmeans on multi GPUs.
    Args:
    -----
    x: data
    nmb_clusters (int): number of clusters
    Returns:
    --------
    list: ids of data in each cluster
    """
    x=np.ascontiguousarray(x)
    n_data, d = x.shape
    ngpus = len(gpu_device)
    assert ngpus > 1

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    clus.seed = seed
    res = [faiss.StandardGpuResources() for i in range(ngpus)]
    flat_config = []
    for i in gpu_device:
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    indexes = [faiss.GpuIndexFlatL2(res[i], d, flat_config[i]) for i in range(ngpus)]
    index = faiss.IndexReplicas()
    for sub_index in indexes:
        index.addIndex(sub_index)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)

    return [int(n[0]) for n in I]

class Kmeans(object):
    """
    Train <k> different k-means clusterings with different 
    random seeds. These will be used to compute close neighbors
    for a given encoding.
    """
    def __init__(self, k, memory_bank, gpu_device=0):
        super().__init__()
        self.k = k
        self.memory_bank = memory_bank
        self.gpu_device = gpu_device

    def compute_clusters(self):
        """
        Performs many k-means clustering.
        
        Args:
            x_data (np.array N * dim): data to cluster
        """
        # data = self.memory_bank.as_tensor()
        data_npy = self.memory_bank.cpu().detach().numpy()
        clusters = self._compute_clusters(data_npy)
        return clusters

    def _compute_clusters(self, data):
        pred_labels = []
        for k_idx, each_k in enumerate(self.k):
            # cluster the data

            if len(self.gpu_device) == 1: # single gpu
                I, _ = run_kmeans(data, each_k, seed=k_idx + DEFAULT_KMEANS_SEED,
                                  gpu_device=self.gpu_device[0])
            else: # multigpu
                I = run_kmeans_multi_gpu(data, each_k, seed=k_idx + DEFAULT_KMEANS_SEED,
                                  gpu_device=self.gpu_device)

            clust_labels = np.asarray(I)
            pred_labels.append(clust_labels)
        pred_labels = np.stack(pred_labels, axis=0)
        pred_labels = torch.from_numpy(pred_labels).long()
        
        return pred_labels