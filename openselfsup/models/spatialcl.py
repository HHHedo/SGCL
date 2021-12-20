import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS
import numpy as np

@MODELS.register_module
class SpatialCL(nn.Module):
    '''Spatial2hCL.
    Part of the code is borrowed from:
        "https://github.com/facebookresearch/moco/blob/master/moco/builder.py".
    '''

    def __init__(self,
                 backbone,
                 neck=None,
                 pretrained=None,
                 loss_lambda=0.5,
                 memory_bank=None, 
                 bag_idxs = None,
                 x_coords = None,
                 y_coords = None,
                 rampup_length = None,
                 similar=None,
                 **kwargs):
        super(SpatialCL, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        # self.head = builder.build_head(head)
        self.memory_bank = builder.build_memory(memory_bank)
        self.init_weights(pretrained=pretrained)
        self.loss_lambda = loss_lambda

        # create the queue
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer('bag_idxs', torch.tensor(bag_idxs))
        self.register_buffer('x_coords', torch.tensor(x_coords))
        self.register_buffer('y_coords', torch.tensor(y_coords))
        self.rampup_length = rampup_length
        self.similar=similar

        ndata = len(bag_idxs)
        # hard mining beta[5,15],20%
        beta = torch.distributions.beta.Beta(torch.tensor([5.]), torch.tensor([15.]))
        samples = beta.sample(sample_shape=torch.Size([1000000]))
        count = torch.histc(samples, bins=100)
        count = torch.nn.functional.interpolate(count.unsqueeze(0).unsqueeze(0), int(0.2*ndata))
        count = torch.nn.functional.normalize(count.squeeze(),dim=0)
        self.pdf = count
        # TODO self.k: the number of neighbour samples; self.dis_threshold, the geometric range
        # self.aux_num: the number of aux anchors
        no_clusters=1000
        no_kmeans=3
        self.dis_threshold=2
        self.aux_num=3
        self.k=4096
        self.nei_k = int(self.k*2/(1+self.aux_num))
        self.kmeans = [no_clusters for _ in range(no_kmeans)]
        cluster_labels = torch.from_numpy(np.random.choice(range(no_clusters),ndata)).long()
        cluster_labels = cluster_labels.unsqueeze(0).repeat(no_kmeans, 1) # (no_kmeans, ndata)
        self.register_buffer(name='cluster', tensor=cluster_labels)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights(init_linear='kaiming')


    def _spatial_ir(self, s_all, bag_idx, x_coord, y_coord):
        # get WSIs (batch_size, ndata)
        chosen_patch_idx = (torch.eq(self.bag_idxs, bag_idx.unsqueeze(-1).expand((-1, self.bag_idxs.shape[0]))))
        # get coords 
        coords = torch.stack((x_coord, y_coord),dim=1).float() # b*2
        coords_bank = torch.stack((self.x_coords, self.y_coords),dim=1).float() # ndata*2
        distance = torch.cdist(coords, coords_bank, p=2) # b*ndata
        distance = (distance < self.dis_threshold) & (0 < distance) 
        pos_idx = chosen_patch_idx * distance # (batch_size, ndata)， 01mask of chosen index
        s_pos_spatial = torch.sum(torch.where(pos_idx, s_all, torch.zeros_like(s_all)), dim=1)  # (batch_size) sum of similarity
        return s_pos_spatial, pos_idx 


    def _simi_ir(self, s_all, q, q_idx, neighbour_idx):
        memory = self.memory_bank.feature_bank.clone().detach()
        # In geometric neighbour find the most similar K aux anchors
        aux_feat, aux_idx = self._get_aux_q(s_all, memory, neighbour_idx)
        all_q = torch.cat((q, aux_feat), dim=0)
        all_idx = torch.cat((q_idx, aux_idx))
        # For each aux_anchor find the top-k samples
        back_nei_dps, back_nei_idxs = self._get_neg_dot_products(all_q, memory, all_idx)
        # Filter by cluster labels multiple times
        all_close_nei_in_back = None
        no_kmeans = self.cluster.size(0)
        with torch.no_grad():
            #sample postive sample
            for each_k_idx in range(no_kmeans):
                curr_close_nei = self._get_close_nei_in_back(each_k_idx, back_nei_idxs, all_idx) # (self.aux_num*batch_size, topk)              
                if all_close_nei_in_back is None:
                    all_close_nei_in_back = curr_close_nei
                else:
                    all_close_nei_in_back = all_close_nei_in_back & curr_close_nei
        # concatenate anchor and auxiliary anchor
        #scatter, dim=1, put src (all_close_nei_in_back(0,1)) by row with the idx (back_nei_idxs)
        pos_idx_scatter = torch.zeros(all_q.shape[0],neighbour_idx.shape[1]).cuda().scatter(1, back_nei_idxs, all_close_nei_in_back.float()) #(bs, ndata)
        batch_pos_idx = (torch.sum(pos_idx_scatter.view(self.aux_num+1, q.shape[0], -1), dim=0)>0).float()
        # if self.aux_num == 1:
        #     batch_idx_sca, q_idx_sca = torch.split(pos_idx_scatter, q.shape[0], dim=0) # 2*(batch_size, ndata)
        #     batch_pos_idx = batch_idx_sca.byte() | q_idx_sca.byte() # (batch_size, ndata)
        # else:
        #     idx_sca = torch.split(pos_idx_scatter, q.shape[0], dim=0) # tuple aux_num * (batch_size, ndata)
        #     batch_pos_idx = idx_sca[0].byte()
        #     for i in idx_sca[1:]:
        #         batch_pos_idx = batch_pos_idx | i.byte()
        pos_logits = s_all * batch_pos_idx # (batch_size, ndata)
        return pos_logits, batch_pos_idx  #exp(logit/T)

    @torch.no_grad()
    def _get_aux_q(self, all_dps, memory, neighbour_idx):
        select_dps = neighbour_idx * all_dps # (batch_size, ndata)
        _, back_idx = torch.topk(select_dps, k=self.aux_num, sorted=False, dim=1) # (batch_size, aux_num)
        back_idx = back_idx.T.reshape(-1)
        q = torch.index_select(memory, 0, back_idx.view(-1))
        return q, back_idx

    @torch.no_grad()
    def _get_neg_dot_products(self, outputs, memory, idx):
        all_dps = torch.einsum('bc,cn->bn', [outputs, memory.T]) # (aux*batch_size, ndata)
        idx_scatter = torch.ones_like(all_dps).scatter(1, idx.unsqueeze(-1), 0) #ignore itself
        all_dps = idx_scatter * all_dps
        back_nei_dps, back_nei_idxs = torch.topk(all_dps, k=self.nei_k , sorted=False, dim=1)
        return back_nei_dps, back_nei_idxs

    @torch.no_grad()
    def _get_close_nei_in_back(self, each_k_idx, back_nei_idxs, idx):
        batch_labels = self.cluster[each_k_idx][idx] # (2*batch_size)
        top_cluster_labels = self.cluster[each_k_idx][back_nei_idxs] # (2*batch_size, topk)
        batch_labels = batch_labels.unsqueeze(1).expand(-1, self.nei_k )
        curr_close_nei = torch.eq(batch_labels, top_cluster_labels) # (2*batch_size, topk)
        return curr_close_nei.byte()
    
    
    def _hard_mining(self, all_dps, outputs, idx, spatial_pos_idx,semantric_pos_idx):
        ir_pos_idx = torch.zeros(outputs.shape[0], self.memory_bank.feature_bank.shape[0]).cuda().scatter(1, idx.view(-1,1), 1)
        pos_idx = ir_pos_idx.byte() | spatial_pos_idx.byte() | semantric_pos_idx.byte()
        # all_dps = torch.exp(torch.einsum('bc,cn->bn', [outputs, self.memory_bank.feature_bank.T])/self.head.temperature) # (batch_size, ndata)
        all_dps = (1 - pos_idx) * all_dps
        batch_size, ndata = all_dps.shape
        back_nei_dps, back_nei_idx = torch.topk(all_dps, k=int(0.2*ndata), sorted=True, dim=1) #(batch_size, 0.2*ndata)
        # mining_dps = torch.narrow(back_nei_dps, 1, 0, self.k)
        select_index = torch.multinomial(self.pdf, self.k)
        mining_dps = back_nei_dps[:, select_index] # (batch_size, k)
        s_mining_negs = torch.sum(mining_dps, dim=1) #(bs , 1) 
        return s_mining_negs

    @torch.no_grad()
    def _cluster_update(self, cluster_labels):
        self.cluster = cluster_labels.cuda()

    def forward_train(self, img, idx, bag_idx, x_coord, y_coord, **kwargs):
        x = self.backbone(img)
        idx = idx.cuda()
        q = self.neck(x)[0]
        q = nn.functional.normalize(q)  # BxC

        
        # Cal all logits(l_all): dot product without /T
        # Cal all similarities(s_all): dot product/T
        l_all = torch.einsum('bc,cn->bn', [q, self.memory_bank.feature_bank.clone().detach().T])
        s_all = torch.exp(l_all/0.2)
        

        # + spatial 
        s_pos_spatial, spatial_pos_idx = self._spatial_ir(s_all, bag_idx, x_coord, y_coord)
        # + semantic
        s_pos_semantic, semantric_pos_idx = self._simi_ir(s_all, q, idx, spatial_pos_idx) # (batch_size, ndata), similarity
        # - hard mining
        s_mining_negs = self._hard_mining(s_all, q, idx, spatial_pos_idx, semantric_pos_idx) #(bs , k=4096), similarity
        # spatial loss
        loss_spatial = -torch.mean(torch.log(s_pos_spatial/(s_pos_spatial + s_mining_negs) + 1e-7)) # (batch_size， 1)
        # semantic loss
        similar_fraction = s_pos_semantic/(s_pos_semantic + s_mining_negs.unsqueeze(-1))  # (batch_size, ndata)
        loss_semantic = -torch.mean(torch.log(torch.sum(nn.functional.normalize(similar_fraction, p=0), dim=1) + 1e-7))
        # InfoNCE
        pos_feat = torch.index_select(self.memory_bank.feature_bank, 0, idx)
        # l_pos_npid_not = torch.einsum('nc,nc->n', [pos_feat, q3]).unsqueeze(-1)
        l_pos_npid = torch.exp(torch.einsum('nc,nc->n', [pos_feat, q])/0.2)
        loss_npid = -torch.mean(torch.log(l_pos_npid/(l_pos_npid + s_mining_negs) + 1e-7))

        # sementic_weight 
        current = np.clip(kwargs['epoch'], 0.0, self.rampup_length)
        phase = 1.0 - current / self.rampup_length
        semnetic_weight = float(np.exp(-5.0 * phase * phase))
        if kwargs['iter']==0 and torch.distributed.get_rank() == 0:
            print('epoch:{}, semantic weight{:.3f}:'.format(kwargs['epoch'], semnetic_weight))
        # gather all losses
        losses = dict()
        losses['loss_contra_single'] = loss_npid * self.loss_lambda
        losses['loss_contra_spatial'] = loss_spatial * semnetic_weight * self.loss_lambda
        losses['loss_contra_sementic'] = loss_semantic  * semnetic_weight * self.loss_lambda
        
        # update memory bank
        with torch.no_grad():
            self.memory_bank.update(idx, q.detach())
        with torch.no_grad():
            # renew self.cluster
            # print('i=',i, 'rank=',torch.distributed.get_rank(),self.similar)
            if kwargs['iter']==0 and torch.distributed.get_rank() == 0 and self.similar:
                print('Fitting K-means with FAISS')
                km = Kmeans(self.kmeans, self.memory_bank.feature_bank, [1,2,3])
                cluster_labels = km.compute_clusters()
                self._cluster_update(cluster_labels)
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