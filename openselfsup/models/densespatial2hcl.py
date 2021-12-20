import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS
import numpy as np

@MODELS.register_module
class DenseSpatial2hCL(nn.Module):
    '''DenseCL.
    Part of the code is borrowed from:
        "https://github.com/facebookresearch/moco/blob/master/moco/builder.py".
    '''

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 loss_lambda_dense=0.5,
                 loss_lambda_spatial=0.5,
                 memory_bank=None, 
                 bag_idxs = None,
                 x_coords = None,
                 y_coords = None,
                 rampup_length = None,
                 similar=None,
                 **kwargs):
        super(DenseSpatial2hCL, self).__init__()
        self.encoder_q = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.encoder_k = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.backbone = self.encoder_q[0]
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        self.head = builder.build_head(head)
        if memory_bank is not None:
            self.memory_bank = builder.build_memory(memory_bank) 
        self.init_weights(pretrained=pretrained)

        self.queue_len = queue_len
        self.momentum = momentum
        self.loss_lambda_dense = loss_lambda_dense
        self.loss_lambda_spatial = loss_lambda_spatial

        # create the queue
        self.register_buffer("queue", torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # create the second queue for dense output
        self.register_buffer("queue2", torch.randn(feat_dim, queue_len))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)
        self.register_buffer("queue2_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer('bag_idxs', torch.tensor(bag_idxs))
        self.register_buffer('x_coords', torch.tensor(x_coords))
        self.register_buffer('y_coords', torch.tensor(y_coords))
        self.register_buffer('coords_bank', torch.stack((self.x_coords, self.y_coords),dim=1).float())
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
        self.kmeans = [no_clusters for _ in range(no_kmeans)]
        cluster_labels = torch.from_numpy(np.random.choice(range(no_clusters),ndata)).long()
        cluster_labels = cluster_labels.unsqueeze(0).repeat(no_kmeans, 1) # (no_kmeans, ndata)
        self.register_buffer(name='cluster', tensor=cluster_labels)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.encoder_q[0].init_weights(pretrained=pretrained)
        self.encoder_q[1].init_weights(init_linear='kaiming')
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue2_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue2[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue2_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def _spatial_ir(self, s_all, bag_idx, x_coord, y_coord):
        # get WSIs (batch_size, ndata)
        chosen_patch_idx = (torch.eq(self.bag_idxs, bag_idx.unsqueeze(-1).expand((-1, self.bag_idxs.shape[0]))))
        # get coords 
        coords = torch.stack((x_coord, y_coord),dim=1).float() # b*2
        # coords_bank = torch.stack((self.x_coords, self.y_coords),dim=1).float() # ndata*2
        distance = torch.cdist(coords, self.coords_bank, p=2) # b*ndata
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
        if self.aux_num == 1:
            batch_idx_sca, q_idx_sca = torch.split(pos_idx_scatter, q.shape[0], dim=0) # 2*(batch_size, ndata)
            batch_pos_idx = batch_idx_sca.byte() | q_idx_sca.byte() # (batch_size, ndata)
        else:
            idx_sca = torch.split(pos_idx_scatter, q.shape[0], dim=0) # tuple aux_num * (batch_size, ndata)
            batch_pos_idx = idx_sca[0].byte()
            for i in idx_sca[1:]:
                batch_pos_idx = batch_pos_idx | i.byte()
        pos_logits = s_all * batch_pos_idx # (batch_size, ndata)
        return pos_logits, batch_pos_idx  #exp(logit/T)

    @torch.no_grad()
    def _get_aux_q(self, all_dps, memory, neighbour_idx):
        select_dps = neighbour_idx * all_dps # (batch_size, ndata)
        _, back_idx = torch.topk(select_dps, k=self.aux_num, sorted=False, dim=1) # (batch_size, aux_num)
        back_idx = back_idx.view(-1)
        q = torch.index_select(memory, 0, back_idx.view(-1))
        return q, back_idx

    @torch.no_grad()
    def _get_neg_dot_products(self, outputs, memory, idx):
        all_dps = torch.einsum('bc,cn->bn', [outputs, memory.T]) # (aux*batch_size, ndata)
        idx_scatter = torch.ones_like(all_dps).scatter(1, idx.unsqueeze(-1), 0) #ignore itself
        all_dps = idx_scatter * all_dps
        back_nei_dps, back_nei_idxs = torch.topk(all_dps, k=self.k, sorted=False, dim=1)
        return back_nei_dps, back_nei_idxs

    @torch.no_grad()
    def _get_close_nei_in_back(self, each_k_idx, back_nei_idxs, idx):
        batch_labels = self.cluster[each_k_idx][idx] # (2*batch_size)
        top_cluster_labels = self.cluster[each_k_idx][back_nei_idxs] # (2*batch_size, topk)
        batch_labels = batch_labels.unsqueeze(1).expand(-1, self.k)
        curr_close_nei = torch.eq(batch_labels, top_cluster_labels) # (2*batch_size, topk)
        return curr_close_nei.byte()
    
    
    def _hard_mining(self, all_dps, outputs, idx, spatial_pos_idx,semantric_pos_idx):
        ir_pos_idx = torch.zeros(outputs.shape[0], self.memory_bank.feature_bank.shape[0]).cuda().scatter(1, idx.view(-1,1), 1)
        pos_idx = ir_pos_idx.byte() | spatial_pos_idx.byte() | semantric_pos_idx.byte()
        # all_dps = torch.einsum('bc,cn->bn', [outputs, self.memory_bank.feature_bank.T]) # (batch_size, ndata)
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
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        im_q = img[:, 0, ...].contiguous()
        im_k = img[:, 1, ...].contiguous()
        # compute query features
        q_b = self.encoder_q[0](im_q) # backbone features
        q, q_grid, q2, q3 = self.encoder_q[1](q_b)  # queries: NxC; NxCxS^2
        q_b = q_b[0]
        q_b = q_b.view(q_b.size(0), q_b.size(1), -1)

        q = nn.functional.normalize(q, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)
        q_grid = nn.functional.normalize(q_grid, dim=1)
        q_b = nn.functional.normalize(q_b, dim=1)
        q3 = nn.functional.normalize(q3, dim=1)


        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k_b = self.encoder_k[0](im_k)
            k, k_grid, k2, _ = self.encoder_k[1](k_b)  # keys: NxC; NxCxS^2
            k_b = k_b[0]
            k_b = k_b.view(k_b.size(0), k_b.size(1), -1)

            k = nn.functional.normalize(k, dim=1)
            k2 = nn.functional.normalize(k2, dim=1)
            k_grid = nn.functional.normalize(k_grid, dim=1)
            k_b = nn.functional.normalize(k_b, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k2 = self._batch_unshuffle_ddp(k2, idx_unshuffle)
            k_grid = self._batch_unshuffle_ddp(k_grid, idx_unshuffle)
            k_b = self._batch_unshuffle_ddp(k_b, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # feat point set sim
        backbone_sim_matrix = torch.matmul(q_b.permute(0, 2, 1), k_b)
        densecl_sim_ind = backbone_sim_matrix.max(dim=2)[1] # NxS^2

        indexed_k_grid = torch.gather(k_grid, 2, densecl_sim_ind.unsqueeze(1).expand(-1, k_grid.size(1), -1)) # NxCxS^2
        densecl_sim_q = (q_grid * indexed_k_grid).sum(1) # NxS^2

        l_pos_dense = densecl_sim_q.view(-1).unsqueeze(-1) # NS^2X1

        q_grid = q_grid.permute(0, 2, 1)
        q_grid = q_grid.reshape(-1, q_grid.size(2))
        l_neg_dense = torch.einsum('nc,ck->nk', [q_grid,
                                            self.queue2.clone().detach()])
        # Cal all logits(l_all): dot product without /T
        # Cal all similarities(s_all): dot product/T
        l_all = torch.einsum('bc,cn->bn', [q3, self.memory_bank.feature_bank.clone().detach().T])
        s_all = torch.exp(l_all/self.head.temperature)

        pos_feat = torch.index_select(self.memory_bank.feature_bank, 0, idx)
        # l_pos_npid_not = torch.einsum('nc,nc->n', [pos_feat, q3]).unsqueeze(-1)
        l_pos_npid = torch.exp(torch.einsum('nc,nc->n', [pos_feat, q3])/self.head.temperature)
        # + spatial 
        s_pos_spatial, spatial_pos_idx = self._spatial_ir(s_all, bag_idx, x_coord, y_coord)
        # + semantic
        s_pos_semantic, semantric_pos_idx = self._simi_ir(s_all, q3, idx, spatial_pos_idx) # (batch_size, ndata), similarity
        # - hard mining
        s_mining_negs = self._hard_mining(s_all, q3, idx, spatial_pos_idx, semantric_pos_idx) #(bs , k=4096), similarity
        # spatial loss
        loss_spatial = -torch.mean(torch.log(s_pos_spatial/(s_pos_spatial + s_mining_negs) + 1e-7)) # (batch_size， 1)
        # semantic loss
        similar_fraction = s_pos_semantic/(s_pos_semantic + s_mining_negs.unsqueeze(-1))  # (batch_size, ndata)
        loss_semantic = -torch.mean(torch.log(torch.sum(nn.functional.normalize(similar_fraction, p=0), dim=1) + 1e-7))
        #NPID
        # loss_npid = self.head(l_pos_npid_not, l_neg_npid)['loss_contra']
        loss_npid = -torch.mean(torch.log(l_pos_npid/(l_pos_npid + s_mining_negs) + 1e-7))
        # InfoNCE
        loss_single = self.head(l_pos, l_neg)['loss_contra']
        # Dense Loss
        loss_dense = self.head(l_pos_dense, l_neg_dense)['loss_contra']

        # sementic_weight 
        current = np.clip(kwargs['epoch'], 0.0, self.rampup_length)
        phase = 1.0 - current / self.rampup_length
        semnetic_weight = float(np.exp(-5.0 * phase * phase))
        if kwargs['iter']==0 and torch.distributed.get_rank() == 0:
            print('epoch:{}, semantic weight{:.3f}:'.format(kwargs['epoch'], semnetic_weight))
        # gather all losses
        losses = dict()
        losses['loss_contra_single'] = loss_single * self.loss_lambda_dense
        losses['loss_contra_dense'] = loss_dense * self.loss_lambda_dense
        losses['loss_contra_spatial'] = loss_spatial * semnetic_weight * self.loss_lambda_spatial
        losses['loss_contra_sementic'] = loss_semantic * semnetic_weight * self.loss_lambda_spatial
        losses['loss_npid'] = loss_npid * self.loss_lambda_spatial
        # for ablation
        # losses['loss_npid'] = loss_npid * 0
        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue2(k2)
        # update memory bank
        with torch.no_grad():
            self.memory_bank.update(idx, q3.detach())
        with torch.no_grad():
            # renew self.cluster
            # print('i=',i, 'rank=',torch.distributed.get_rank(),self.similar)
            if kwargs['iter']==0 and torch.distributed.get_rank() == 0 and self.similar:
                print('Fitting K-means with FAISS')
                km = Kmeans(self.kmeans, self.memory_bank.feature_bank, [1,2,3])
                cluster_labels = km.compute_clusters()
                self._cluster_update(cluster_labels)
        return losses
        #test

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