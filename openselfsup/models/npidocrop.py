import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS


@MODELS.register_module
class NpidOcrop(nn.Module):
    '''SpatialCL.
    Part of the code is borrowed from:
        "https://github.com/facebookresearch/moco/blob/master/moco/builder.py".
    '''

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 loss_lambda=0.5,
                 memory_bank=None, 
                 memory_bank_b=None, 
                 memory_bank_c=None, 
                 bag_idxs = None,
                 x_coords = None,
                 y_coords = None,
                 rampup_length = None,
                 similar=None,
                 num_crops=[1, 3],
                #  temperature=0.07,
                 no_clusters=1000,
                 no_kmeans=3,
                 dis_threshold=2,
                 aux_num=1,
                 k=4096,
                 nei_k = 4096,
                 **kwargs):
        super(NpidOcrop, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)
        self.memory_bank = builder.build_memory(memory_bank)
        self.memory_bank_c = builder.build_memory(memory_bank_c)
        self.init_weights(pretrained=pretrained)
        self.loss_lambda = loss_lambda
        self.num_crops = num_crops
        self.k = k
        # create the queue

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights(init_linear='kaiming')


    def forward_train(self, img, img2, repeat_idx, idx, bag_idx, x_coord, y_coord, **kwargs):
        assert isinstance(img2, list)
        img=img2
        # multi-res forward passes
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
               torch.tensor([i.shape[-1] for i in img]),
                return_counts=True)[1], 0)
        start_idx = 0
        feature = []
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(img[start_idx:end_idx]))
            feature.append(_out)
            start_idx = end_idx
        idx = idx.cuda()
        repeat_idx = torch.cat(repeat_idx).cuda()
        q = self.neck(feature)[0]
        q = nn.functional.normalize(q)  # BxC
        bs, feat_dim = q.shape[:2]
        q_sslp = q[:int(bs/2)]
        q_l = q[int(bs/2):]
        bs_g, feat_dim = q_sslp.shape[:2]
        bs_l, _ = q_l.shape[:2]

        neg_idx = self.memory_bank.multinomial.draw(bs_g * self.k)
        pos_feat = torch.index_select(self.memory_bank.feature_bank, 0,
                                      idx)  # BXC
        neg_feat = torch.index_select(self.memory_bank.feature_bank, 0,
                                      neg_idx).view(bs_g, self.k,
                                                    feat_dim)  # BxKxC
        pos_logits_g = torch.einsum('nc,nc->n',
                                  [pos_feat, q_sslp]).unsqueeze(-1)
        neg_logits_g = torch.bmm(neg_feat,  q_sslp.unsqueeze(2)).squeeze(2)

        pos_feat_l = torch.index_select(self.memory_bank_c.feature_bank, 0,
                                    idx)  # BXC
        neg_feat_l = torch.index_select(self.memory_bank_c.feature_bank, 0,
                                      neg_idx).view(bs_l, int(self.k/self.num_crops[1]),
                                                    feat_dim)  
        pos_logits_l = torch.einsum('nc,nc->n',
                                  [pos_feat_l, q_l]).unsqueeze(-1)
        neg_logits_l = torch.bmm(neg_feat_l, q_l.unsqueeze(2)).squeeze(2)

        loss_npid = self.head(pos_logits_g, neg_logits_g)['loss_contra']
        loss_mcrop = self.head(pos_logits_l, neg_logits_l)['loss_contra']
        # w1 = self.loss_lambda1/(self.loss_lambda1+self.loss_lambda2 * self.warm_up)
        # w2 = 1-w1

        # gather all losses
        losses = dict()
        losses['loss_contra_single'] = loss_npid * self.loss_lambda
        losses['loss_contra_mcrop'] = loss_mcrop * self.loss_lambda
            
        with torch.no_grad():
            if kwargs['iter']%50==0 and torch.distributed.get_rank() == 0:
                # sp_num = spatial_pos_idx.float().sum(1).mean()
                # sp_var, sp_m = torch.var_mean(torch.sum((spatial_pos_idx.byte())*s_all, dim=1)/spatial_pos_idx.float().sum(1))
                # se_num = semantric_pos_idx.float().sum(1).mean()
                # se_var, se_m = torch.var_mean(torch.sum((semantric_pos_idx.byte())*s_all, dim=1)/semantric_pos_idx.float().sum(1))
                # hard_neg =  ((s_whole - s_spatial_numerator)/self.nei_k).mean()
                g_v_p, g_m_p = torch.var_mean(torch.exp(pos_logits_g/0.07))
                g_v_n, g_m_n  = torch.var_mean(torch.exp(neg_logits_g/0.07))
                l_v_p, l_m_p = torch.var_mean(torch.exp(pos_logits_l/0.07))
                l_v_n, l_m_n  = torch.var_mean(torch.exp(neg_logits_l/0.07))
                print(' IR:{:.1e}+-{:.1e}/{:.1e}+-{:.1e}\
                     Mcrop:{:.1e}+-{:.1e}/{:.1e}+-{:.1e}\
                         IR:{:.1e}/{:.1e}\
                     Mcrop:{:.1e}/{:.1e}'
                .format(g_m_p , g_v_p, g_m_n , g_v_n,
                l_m_p , l_v_p, l_m_n , l_v_n,
                torch.exp(pos_logits_g/0.2).mean(),
                torch.exp(neg_logits_g/0.2).mean(),
                torch.exp(pos_logits_l/0.2).mean(),
                torch.exp(neg_logits_l/0.2).mean()))
            # update memory bank
            self.memory_bank.update(idx, q_sslp.detach())
            self.memory_bank_c.update(idx, q_l.detach())
        return losses


    def forward_test(self, img, **kwargs):
        im_q = img.contiguous()
        # compute query features
        #_, q_grid, _ = self.encoder_q(im_q)
        q_grid = self.backbone(im_q)[0]
        q_grid = q_grid.view(q_grid.size(0), q_grid.size(1), -1)
        q_grid = nn.functional.normalize(q_grid, dim=1)
        return None, q_grid, None

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


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