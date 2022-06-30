import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS
import numpy as np

@MODELS.register_module
class SpatialCLMcrop(nn.Module):
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
                 memory_bank_tr=None, 
                 memory_bank_tl=None, 
                 memory_bank_br=None, 
                 memory_bank_bl=None, 
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
                 nei_k=4096,
                 **kwargs):
        super(SpatialCLMcrop, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.pool =  nn.AdaptiveAvgPool2d((1, 1))
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)
        # self.proj = nn.Linear(512, 128)
        self.memory_bank = builder.build_memory(memory_bank)
        self.memory_bank_tl = builder.build_memory(memory_bank_tl)
        self.memory_bank_tr = builder.build_memory(memory_bank_tr)
        self.memory_bank_bl = builder.build_memory(memory_bank_bl)
        self.memory_bank_br = builder.build_memory(memory_bank_br)
        self.memory_bank_b = builder.build_memory(memory_bank_b)
        self.init_weights(pretrained=pretrained)
        self.loss_lambda = loss_lambda
        self.T = 0.07
        self.num_crops = num_crops

        self.register_buffer('bag_idxs', torch.tensor(bag_idxs))
        self.register_buffer('coords_bank', torch.stack((torch.tensor(x_coords),torch.tensor(y_coords)),dim=1).float())
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
        # no_clusters=1000
        # no_kmeans=3
        self.dis_threshold=dis_threshold
        self.aux_num=aux_num
        self.k=k
        self.nei_k=nei_k
        #init clustering
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
        # coords_bank = torch.stack((self.x_coords, self.y_coords),dim=1).float() # ndata*2
        # distance = torch.cdist(coords, coords_bank, p=2) # b*ndata
        distance = torch.cdist(coords, self.coords_bank, p=2) # b*ndata
        distance = (distance < self.dis_threshold) & (0 < distance) 
        pos_idx = chosen_patch_idx * distance # (batch_size, ndata)， 01mask of chosen index
        s_pos_spatial = torch.sum(torch.where(pos_idx, s_all, torch.zeros_like(s_all)), dim=1)  # (batch_size) sum of similarity
        return s_pos_spatial, pos_idx 


    def _simi_ir(self, s_all, q, q_idx, neighbour_idx):
        # memory = self.memory_bank.feature_bank.clone().detach()
        memory = self.memory_bank.feature_bank.clone().detach()
        # In geometric neighbour find the most similar K aux anchors
        aux_feat, aux_idx = self._get_aux_q(s_all, memory, neighbour_idx)
        all_q = torch.cat((q, aux_feat), dim=0)
        all_idx = torch.cat((q_idx, aux_idx))
        # For each aux_anchor find the top-k samples, kNN
        back_nei_dps, back_nei_idxs = self._get_neg_dot_products(all_q, memory, all_idx)
        # Filter by cluster labels multiple times
        all_close_nei_in_back = None
        no_kmeans = self.cluster.size(0)
        with torch.no_grad():
            #sample postive sample, k-means
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
        all_dps = torch.einsum('bc,nc->bn', [outputs, memory]) # (aux*batch_size, ndata)
        idx_scatter = torch.ones_like(all_dps).scatter(1, idx.unsqueeze(-1), 0) #ignore itself
        all_dps = idx_scatter * all_dps
        back_nei_dps, back_nei_idxs = torch.topk(all_dps, k=self.nei_k , sorted=False, dim=1)
        return back_nei_dps, back_nei_idxs

    @torch.no_grad()
    def _get_close_nei_in_back(self, each_k_idx, back_nei_idxs, idx):
        batch_labels = self.cluster[each_k_idx][idx] # (2*batch_size)
        top_cluster_labels = self.cluster[each_k_idx][back_nei_idxs] # (2*batch_size, topk)
        batch_labels = batch_labels.unsqueeze(1).expand(-1, self.nei_k)
        curr_close_nei = torch.eq(batch_labels, top_cluster_labels) # (2*batch_size, topk)
        return curr_close_nei.byte()
    
    
    def _hard_mining(self, all_dps, all_dps_b, outputs, idx, spatial_pos_idx, semantric_pos_idx):
        ir_pos_idx = torch.zeros(outputs.shape[0], self.memory_bank.feature_bank.shape[0]).cuda().scatter(1, idx.view(-1,1), 1)
        pos_idx_npid = ir_pos_idx.byte() 
        pos_idx_spatial = ir_pos_idx.byte() | spatial_pos_idx.byte()
        pos_idx_semantic = ir_pos_idx.byte() | spatial_pos_idx.byte() | semantric_pos_idx.byte()
        # all_dps = torch.exp(torch.einsum('bc,cn->bn', [outputs, self.memory_bank.feature_bank.T])/self.head.temperature) # (batch_size, ndata)

        # #info
        # all_dps1 = (1 - pos_idx_semantic) * all_dps
        # batch_size, ndata = all_dps.shape
        # back_nei_dps, back_nei_idx = torch.topk(all_dps1, k=int(0.2*ndata), sorted=True, dim=1) #(batch_size, 0.2*ndata)
        # # mining_dps = torch.narrow(back_nei_dps, 1, 0, self.k)
        # select_index = torch.multinomial(self.pdf, self.k)
        # mining_dps_npid = back_nei_dps[:, select_index] # (batch_size, k)
        # # s_mining_negs_npid = torch.sum(torch.exp(mining_dps_npid/self.T), dim=1) #(bs , 1) 

        # #spatial
        # all_dps2 = (1 - pos_idx_semantic) * all_dps
        # batch_size, ndata = all_dps.shape
        # back_nei_dps, back_nei_idx = torch.topk(all_dps2, k=int(0.2*ndata), sorted=True, dim=1) #(batch_size, 0.2*ndata)
        # # mining_dps = torch.narrow(back_nei_dps, 1, 0, self.k)
        # select_index = torch.multinomial(self.pdf, self.k)
        # mining_dps_spatial = back_nei_dps[:, select_index] # (batch_size, k)
        # s_mining_negs_spatial = torch.sum(torch.exp(mining_dps_spatial/self.T), dim=1) #(bs ) 

        #semantic
        with torch.no_grad():
            all_dps3 = (1 - pos_idx_semantic) * all_dps_b
            # all_dps4 = (1 - pos_idx_semantic) * all_dps
            batch_size, ndata = all_dps.shape
            _, back_nei_idx = torch.topk(all_dps3, k=int(0.2*ndata), sorted=True, dim=1) #(batch_size, 0.2*ndata)
            back_nei_dps = torch.gather(all_dps,1,back_nei_idx)
            # mining_dps = torch.narrow(back_nei_dps, 1, 0, self.k)
            select_index = torch.multinomial(self.pdf, self.nei_k)
            mining_dps_semantic = back_nei_dps[:, select_index] # (batch_size, k)
        s_mining_negs_semantic = torch.sum(torch.exp(mining_dps_semantic/self.T), dim=1) #(bs ) 
        s_spatial_sementic = torch.sum(torch.exp(((spatial_pos_idx.byte()|semantric_pos_idx.byte())*all_dps)/self.T),dim=1)
        s_whole = s_spatial_sementic + s_mining_negs_semantic
        return s_whole

    @torch.no_grad()
    def _cluster_update(self, cluster_labels):
        self.cluster = cluster_labels.cuda()

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
        # x = self.backbone(img)
        x = self.pool(feature[0][0]) # global bs*512
        idx = idx.cuda()
        repeat_idx = torch.cat(repeat_idx).cuda()
        # q, q_l = self.neck(feature)
        q = self.neck(feature)[0]
        # bs, feat_dim = q.shape[:2]
        q_norm = nn.functional.normalize(q)  # BxC
        # q_l = nn.functional.normalize(q_l)  # BxC
        x = nn.functional.normalize(x.view(x.size(0), -1))
        q_sslp = q_norm[:x.shape[0]]
        # q_g = q_l[:x.shape[0]]
        q_l = q_norm[x.shape[0]:]
        bs_g, feat_dim = q_sslp.shape[:2]
        bs_l, _ = q_l.shape[:2]
        # bs, feat_dim = q.shape[:2]
        
        # Cal all logits(l_all): dot product without /T
        # Cal all similarities(s_all): dot product/T
        
        d_all = torch.einsum('bc,nc->bn', [q_sslp, self.memory_bank.feature_bank.clone().detach()])
        s_all = torch.exp(d_all/self.T)
        with torch.no_grad():
            d_all_b = torch.einsum('bc,nc->bn', [x, self.memory_bank_b.feature_bank.clone().detach()])
            s_all_b = torch.exp(d_all_b/self.T)
            # + spatial 
            _, spatial_pos_idx = self._spatial_ir(s_all, bag_idx, x_coord, y_coord)
            # + semantic
            # _, semantric_pos_idx = self._simi_ir(s_all, q, idx, spatial_pos_idx) # (batch_size, ndata), similarity
            _, semantric_pos_idx = self._simi_ir(s_all, q_sslp, idx, spatial_pos_idx) # (batch_size, ndata), similarity
        # - hard mining
        # - hard mining
        s_whole = self._hard_mining(d_all, d_all_b, q_sslp, idx, spatial_pos_idx, semantric_pos_idx) #(bs , k=4096), similarity
        s_spatial_numerator = torch.sum((semantric_pos_idx.byte()|spatial_pos_idx.byte())*s_all, dim=1)
        loss_spatial = -torch.mean(torch.log(s_spatial_numerator/s_whole + 1e-7))

        neg_idx = self.memory_bank.multinomial.draw(bs_g * self.k)
        # neg_idx = neg_idx.view(bs, -1)
        # while True:
        #     wrong = (neg_idx == idx.view(-1, 1))
        #     if wrong.sum().item() > 0:
        #         neg_idx[wrong] = self.memory_bank.multinomial.draw(
        #             wrong.sum().item())
        #     else:
        #         break
        # neg_idx = neg_idx.flatten()
        pos_feat = torch.index_select(self.memory_bank.feature_bank, 0,
                                      idx)  # BXC
        neg_feat = torch.index_select(self.memory_bank.feature_bank, 0,
                                      neg_idx).view(bs_g, self.k,
                                                    feat_dim)  # BxKxC
        pos_logits_g = torch.einsum('nc,nc->n',
                                  [pos_feat, q_sslp]).unsqueeze(-1)
        neg_logits_g = torch.bmm(neg_feat,  q_sslp.unsqueeze(2)).squeeze(2)

        # pos_feat2 = torch.index_select(self.memory_bank2.feature_bank, 0,
        #                               idx)  # BXC
        # neg_feat2 = torch.index_select(self.memory_bank2.feature_bank, 0,
        #                               neg_idx).view(bs_g, self.k,
        #                                             feat_dim)  # BxKxC
        # pos_feat2 =  torch.index_select(self.memory_bank.feature_bank, 0,
        #                               repeat_idx[x.shape[0]:])  # 4BXC

        # pos_feat2 =  torch.cat([pos_feat for _ in range(self.num_crops[1])])
        # neg_feat2 =  torch.cat([neg_feat for _ in range(self.num_crops[1])])
        # pos_logits_g2l = torch.einsum('nc,nc->n',
        #                           [pos_feat2, q_l]).unsqueeze(-1)
        # neg_logits_g2l = torch.bmm(neg_feat2,  q_l.unsqueeze(2)).squeeze(2)

        ################
        # q_l2g = self.proj(torch.cat([q[bs_g:2*bs_g],  q[2*bs_g:3*bs_g], q[3*bs_g:4*bs_g], q[4*bs_g:]], 1)) #b*4c --b*c
        # q_l2g = nn.functional.normalize(q_l2g)
        # pos_logits_g2l = torch.einsum('nc,nc->n',
        #                           [q_sslp.detach(), q_l2g]).unsqueeze(-1)
        # neg_logits_g2l = torch.bmm(neg_feat,  q_l2g.unsqueeze(2)).squeeze(2)

        pos_feat_tl = torch.index_select(self.memory_bank_tl.feature_bank, 0,
                                    idx)  # BXC
        pos_feat_tr = torch.index_select(self.memory_bank_tr.feature_bank, 0,
                                    idx)  # BXC
        pos_feat_bl = torch.index_select(self.memory_bank_bl.feature_bank, 0,
                                    idx)  # BXC
        pos_feat_br = torch.index_select(self.memory_bank_br.feature_bank, 0,
                                    idx)  # BXC
        pos_feat_l = torch.cat([pos_feat_tl, pos_feat_tr, pos_feat_bl, pos_feat_br]) # 4BxKxC
        
        neg_feat_tl = torch.index_select(self.memory_bank_tl.feature_bank, 0,
                                      neg_idx).view(bs_l, int(self.k/self.num_crops[1]),
                                                    feat_dim)  # 4BxK/4xC
        neg_feat_tr = torch.index_select(self.memory_bank_tr.feature_bank, 0,
                                      neg_idx).view(bs_l, int(self.k/self.num_crops[1]),
                                                    feat_dim)  
        neg_feat_bl = torch.index_select(self.memory_bank_bl.feature_bank, 0,
                                      neg_idx).view(bs_l, int(self.k/self.num_crops[1]),
                                                    feat_dim)  
        neg_feat_br = torch.index_select(self.memory_bank_br.feature_bank, 0,
                                      neg_idx).view(bs_l, int(self.k/self.num_crops[1]),
                                                    feat_dim)  
        neg_feat_l = torch.cat([neg_feat_tl, neg_feat_tr, neg_feat_bl, neg_feat_br],1) # 4BxKxC
        pos_logits_l2l = torch.einsum('nc,nc->n',
                                  [pos_feat_l, q_l]).unsqueeze(-1)
        neg_logits_l2l = torch.bmm(neg_feat_l, q_l.unsqueeze(2)).squeeze(2)




        loss_npid = self.head(pos_logits_g, neg_logits_g)['loss_contra']
        # loss_g2l = self.head(pos_logits_g2l, neg_logits_g2l)['loss_contra']
        loss_mcrop = self.head(pos_logits_l2l, neg_logits_l2l)['loss_contra']
        # w1 = self.loss_lambda1/(self.loss_lambda1+self.loss_lambda2 * self.warm_up)
        # w2 = 1-w1

        # sementic_weight 
        current = np.clip(kwargs['epoch'], 0.0, self.rampup_length)
        phase = 1.0 - current / self.rampup_length
        semnetic_weight = float(np.exp(-5.0 * phase * phase))
        # npid_weight = 1/(1 + semnetic_weight + semnetic_weight)
        # semnetic_weight = semnetic_weight/(1 + semnetic_weight + semnetic_weight)
        if kwargs['iter']==0 and torch.distributed.get_rank() == 0:
            print('epoch:{}, semantic weight{:.3f}:'.format(kwargs['epoch'], semnetic_weight))
        # gather all losses
        losses = dict()
        # losses['loss_contra_single'] = loss_npid * npid_weight
        # losses['loss_contra_spatial'] = loss_spatial * semnetic_weight 
        # losses['loss_contra_sementic'] = loss_semantic  * semnetic_weight
        losses['loss_contra_single'] = loss_npid 
        losses['loss_contra_spatial'] = loss_spatial * semnetic_weight 
        losses['loss_contra_mcrop'] = loss_mcrop 
        # losses['loss_contra_single2'] = loss_g2l 

        # losses['loss_contra_sementic'] = loss_semantic  * semnetic_weight 
            
        with torch.no_grad():
            if kwargs['iter']%50==0 and torch.distributed.get_rank() == 0 and self.similar:
                sp_num = spatial_pos_idx.float().sum(1).mean()
                sp_var, sp_m = torch.var_mean(torch.sum((spatial_pos_idx.byte())*s_all, dim=1)/spatial_pos_idx.float().sum(1))
                se_num = semantric_pos_idx.float().sum(1).mean()
                se_var, se_m = torch.var_mean(torch.sum((semantric_pos_idx.byte())*s_all, dim=1)/semantric_pos_idx.float().sum(1))
                hard_neg =  ((s_whole - s_spatial_numerator)/self.nei_k).mean()
                g_v_p, g_m_p = torch.var_mean(torch.exp(pos_logits_g/self.T))
                g_v_n, g_m_n  = torch.var_mean(torch.exp(neg_logits_g/self.T))
                l_v_p, l_m_p = torch.var_mean(torch.exp(pos_logits_l2l/self.T))
                l_v_n, l_m_n  = torch.var_mean(torch.exp(neg_logits_l2l/self.T))
                print('Spatials:{:.0f}*{:.1e}+-{:.1e}={:.1e}, Sementics:{:.0f}*{:.1e}+-{:.1e}={:.1e}, HN:{:.1e}, IR:{:.1e}+-{:.1e}/{:.1e}+-{:.1e}, Mcrop:{:.1e}+-{:.1e}/{:.1e}+-{:.1e}'
                .format( sp_num,  sp_m, sp_var, torch.sum((spatial_pos_idx.byte())*s_all, dim=1).mean(),
                se_num,  se_m, se_var, torch.sum((semantric_pos_idx.byte())*s_all, dim=1).mean(), 
                hard_neg,
                g_m_p , g_v_p, g_m_n , g_v_n,
                l_m_p , l_v_p, l_m_n , l_v_n))
            # renew self.cluster
            # print('i=',i, 'rank=',torch.distributed.get_rank(),self.similar)
            if kwargs['iter']==0 and torch.distributed.get_rank() == 0 and self.similar:
                print('Fitting K-means with FAISS')
                km = Kmeans(self.kmeans, self.memory_bank.feature_bank, [0,1,2,3])
                cluster_labels = km.compute_clusters()
                self._cluster_update(cluster_labels)
            # update memory bank
            self.memory_bank.update(idx, q_sslp.detach())
            # self.memory_bank2.update(idx, q_g.detach())
            self.memory_bank_b.update(idx, x.detach())
            self.memory_bank_tl.update(idx, q_l[:bs_g].detach())
            self.memory_bank_tr.update(idx, q_l[bs_g:2*bs_g].detach())
            self.memory_bank_bl.update(idx, q_l[2*bs_g:3*bs_g].detach())
            self.memory_bank_br.update(idx, q_l[3*bs_g:].detach())
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



@MODELS.register_module
class SpatialCLMcrop2(nn.Module):
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
                 memory_bank_l=None, 
                 memory_bank_bl=None, 
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
        super(SpatialCLMcrop2, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.pool =  nn.AdaptiveAvgPool2d((1, 1))
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)
        self.memory_bank = builder.build_memory(memory_bank)
        self.memory_bank_l = builder.build_memory(memory_bank_l)
        self.memory_bank_b = builder.build_memory(memory_bank_b)
        self.memory_bank_bl = builder.build_memory(memory_bank_bl)
        self.init_weights(pretrained=pretrained)
        self.loss_lambda = loss_lambda
        self.T = 0.07
        self.num_crops = num_crops
        # create the queue
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer('bag_idxs', torch.tensor(bag_idxs)) # [681486]
        # self.register_buffer('x_coords', torch.tensor(x_coords))
        # self.register_buffer('y_coords', torch.tensor(y_coords))
        self.register_buffer('coords_bank', torch.stack((torch.tensor(x_coords),torch.tensor(y_coords)),dim=1).float()) #[681486 ,2]
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
        # no_clusters=1000
        # no_kmeans=3
        self.dis_threshold=dis_threshold
        self.aux_num=aux_num
        self.k=k
        self.nei_k=nei_k
        #init clustering
        self.kmeans = [no_clusters for _ in range(no_kmeans)]
        cluster_labels = torch.from_numpy(np.random.choice(range(no_clusters),ndata)).long()
        cluster_labels = cluster_labels.unsqueeze(0).repeat(no_kmeans, 1) # (no_kmeans, ndata)
        self.register_buffer(name='cluster', tensor=cluster_labels) #[3, 681486]
        self.register_local( bag_idxs, x_coords, y_coords, cluster_labels)


    def register_local(self, bag_idxs, x_coords, y_coords, cluster_labels):
        bag_idxs = torch.tensor(bag_idxs)
        l_bag_idxs = torch.cat([bag_idxs for _ in range(self.num_crops[1])])
        self.register_buffer('bag_idxs_l', l_bag_idxs)
        x_coords_tl = torch.tensor(x_coords)+0.25
        y_coords_tl = torch.tensor(y_coords)+0.25
        x_coords_tr = torch.tensor(x_coords)+0.75
        y_coords_tr = torch.tensor(y_coords)+0.25
        x_coords_bl = torch.tensor(x_coords)+0.25
        y_coords_bl = torch.tensor(y_coords)+0.75
        x_coords_br = torch.tensor(x_coords)+0.75
        y_coords_br = torch.tensor(y_coords)+0.25
        tl =  torch.stack((x_coords_tl,y_coords_tl),dim=1)
        tr = torch.stack((x_coords_tr,y_coords_tr),dim=1)
        bl =  torch.stack((x_coords_bl,y_coords_bl),dim=1)
        br = torch.stack((x_coords_br,y_coords_br),dim=1)
        coors_bank_l = torch.cat((tl, tr, bl ,br), dim=0)
        self.register_buffer('coords_bank_l', coors_bank_l)
        cluster_labels_l = torch.cat([cluster_labels for _ in range(self.num_crops[1])], dim=1)
        self.register_buffer(name='cluster_l', tensor=cluster_labels_l) #[3, *4]
    

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
        # coords_bank = torch.stack((self.x_coords, self.y_coords),dim=1).float() # ndata*2
        # distance = torch.cdist(coords, coords_bank, p=2) # b*ndata
        distance = torch.cdist(coords, self.coords_bank, p=2) # b*ndata
        distance = (distance < self.dis_threshold) & (0 < distance) 
        pos_idx = chosen_patch_idx * distance # (batch_size, ndata)， 01mask of chosen index
        s_pos_spatial = torch.sum(torch.where(pos_idx, s_all, torch.zeros_like(s_all)), dim=1)  # (batch_size) sum of similarity
        return s_pos_spatial, pos_idx 


    def _simi_ir(self, s_all, q, q_idx, neighbour_idx):
        # memory = self.memory_bank.feature_bank.clone().detach()
        memory = self.memory_bank.feature_bank.clone().detach()
        # In geometric neighbour find the most similar K aux anchors
        aux_feat, aux_idx = self._get_aux_q(s_all, memory, neighbour_idx)
        all_q = torch.cat((q, aux_feat), dim=0)
        all_idx = torch.cat((q_idx, aux_idx))
        # For each aux_anchor find the top-k samples, kNN
        back_nei_dps, back_nei_idxs = self._get_neg_dot_products(all_q, memory, all_idx)
        # Filter by cluster labels multiple times
        all_close_nei_in_back = None
        no_kmeans = self.cluster.size(0)
        with torch.no_grad():
            #sample postive sample, k-means
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
        all_dps = torch.einsum('bc,nc->bn', [outputs, memory]) # (aux*batch_size, ndata)
        idx_scatter = torch.ones_like(all_dps).scatter(1, idx.unsqueeze(-1), 0) #ignore itself
        all_dps = idx_scatter * all_dps
        back_nei_dps, back_nei_idxs = torch.topk(all_dps, k=self.nei_k , sorted=False, dim=1)
        return back_nei_dps, back_nei_idxs

    @torch.no_grad()
    def _get_close_nei_in_back(self, each_k_idx, back_nei_idxs, idx):
        batch_labels = self.cluster[each_k_idx][idx] # (2*batch_size)
        top_cluster_labels = self.cluster[each_k_idx][back_nei_idxs] # (2*batch_size, topk)
        batch_labels = batch_labels.unsqueeze(1).expand(-1, self.nei_k)
        curr_close_nei = torch.eq(batch_labels, top_cluster_labels) # (2*batch_size, topk)
        return curr_close_nei.byte()
    
    
    def _hard_mining(self, all_dps, all_dps_b, outputs, idx, spatial_pos_idx, semantric_pos_idx):
        ir_pos_idx = torch.zeros(outputs.shape[0], self.memory_bank.feature_bank.shape[0]).cuda().scatter(1, idx.view(-1,1), 1)
        pos_idx_npid = ir_pos_idx.byte() 
        pos_idx_spatial = ir_pos_idx.byte() | spatial_pos_idx.byte()
        pos_idx_semantic = ir_pos_idx.byte() | spatial_pos_idx.byte() | semantric_pos_idx.byte()
        # all_dps = torch.exp(torch.einsum('bc,cn->bn', [outputs, self.memory_bank.feature_bank.T])/self.head.temperature) # (batch_size, ndata)

        # #info
        # all_dps1 = (1 - pos_idx_semantic) * all_dps
        # batch_size, ndata = all_dps.shape
        # back_nei_dps, back_nei_idx = torch.topk(all_dps1, k=int(0.2*ndata), sorted=True, dim=1) #(batch_size, 0.2*ndata)
        # # mining_dps = torch.narrow(back_nei_dps, 1, 0, self.k)
        # select_index = torch.multinomial(self.pdf, self.k)
        # mining_dps_npid = back_nei_dps[:, select_index] # (batch_size, k)
        # # s_mining_negs_npid = torch.sum(torch.exp(mining_dps_npid/self.T), dim=1) #(bs , 1) 

        # #spatial
        # all_dps2 = (1 - pos_idx_semantic) * all_dps
        # batch_size, ndata = all_dps.shape
        # back_nei_dps, back_nei_idx = torch.topk(all_dps2, k=int(0.2*ndata), sorted=True, dim=1) #(batch_size, 0.2*ndata)
        # # mining_dps = torch.narrow(back_nei_dps, 1, 0, self.k)
        # select_index = torch.multinomial(self.pdf, self.k)
        # mining_dps_spatial = back_nei_dps[:, select_index] # (batch_size, k)
        # s_mining_negs_spatial = torch.sum(torch.exp(mining_dps_spatial/self.T), dim=1) #(bs ) 

        #semantic
        with torch.no_grad():
            all_dps3 = (1 - pos_idx_semantic) * all_dps_b
            # all_dps4 = (1 - pos_idx_semantic) * all_dps
            batch_size, ndata = all_dps.shape
            _, back_nei_idx = torch.topk(all_dps3, k=int(0.2*ndata), sorted=True, dim=1) #(batch_size, 0.2*ndata)
            back_nei_dps = torch.gather(all_dps,1,back_nei_idx)
            # mining_dps = torch.narrow(back_nei_dps, 1, 0, self.k)
            select_index = torch.multinomial(self.pdf, self.nei_k)
            mining_dps_semantic = back_nei_dps[:, select_index] # (batch_size, k)
        s_mining_negs_semantic = torch.sum(torch.exp(mining_dps_semantic/self.T), dim=1) #(bs ) 
        s_spatial_sementic = torch.sum(torch.exp(((spatial_pos_idx.byte()|semantric_pos_idx.byte())*all_dps)/self.T),dim=1)
        s_whole = s_spatial_sementic + s_mining_negs_semantic
        return s_whole

    @torch.no_grad()
    def _cluster_update(self, cluster_labels):
        self.cluster = cluster_labels.cuda()

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
        # x = self.backbone(img)
        x = self.pool(feature[0][0]) # global bs*512
        idx = idx.cuda()
        repeat_idx = torch.cat(repeat_idx).cuda()
        q, q_l = self.neck(feature)
        # q = self.neck(feature)[0]
        # bs, feat_dim = q.shape[:2]
        q = nn.functional.normalize(q)  # BxC
        q_l = nn.functional.normalize(q_l)  # BxC
        x = nn.functional.normalize(x.view(x.size(0), -1))
        q_sslp = q[:x.shape[0]]
        q_l = q_l[x.shape[0]:]
        bs_g, feat_dim = q_sslp.shape[:2]
        bs_l, _ = q_l.shape[:2]
        # bs, feat_dim = q.shape[:2]
        
        # Cal all logits(l_all): dot product without /T
        # Cal all similarities(s_all): dot product/T
        
        d_all = torch.einsum('bc,nc->bn', [q_sslp, self.memory_bank.feature_bank.clone().detach()])
        s_all = torch.exp(d_all/self.T)
        with torch.no_grad():
            d_all_b = torch.einsum('bc,nc->bn', [x, self.memory_bank_b.feature_bank.clone().detach()])
            s_all_b = torch.exp(d_all_b/self.T)
            # + spatial 
            _, spatial_pos_idx = self._spatial_ir(s_all, bag_idx, x_coord, y_coord)
            # + semantic
            # _, semantric_pos_idx = self._simi_ir(s_all, q, idx, spatial_pos_idx) # (batch_size, ndata), similarity
            _, semantric_pos_idx = self._simi_ir(s_all, q_sslp, idx, spatial_pos_idx) # (batch_size, ndata), similarity
        # - hard mining
        # - hard mining
        s_whole = self._hard_mining(d_all, d_all_b, q_sslp, idx, spatial_pos_idx, semantric_pos_idx) #(bs , k=4096), similarity
        s_spatial_numerator = torch.sum((semantric_pos_idx.byte()|spatial_pos_idx.byte())*s_all, dim=1)
        loss_spatial = -torch.mean(torch.log(s_spatial_numerator/s_whole + 1e-7))


        d_all = torch.einsum('bc,nc->bn', [q_l, self.memory_bank_l.feature_bank.clone().detach()])
        s_all = torch.exp(d_all/self.T)
        with torch.no_grad():
            d_all_b = torch.einsum('bc,nc->bn', [x, self.memory_bank_bl.feature_bank.clone().detach()])
            s_all_b = torch.exp(d_all_b/self.T)
            # + spatial 
            _, spatial_pos_idx = self._spatial_ir(s_all, bag_idx, x_coord, y_coord)
            # + semantic
            # _, semantric_pos_idx = self._simi_ir(s_all, q, idx, spatial_pos_idx) # (batch_size, ndata), similarity
            _, semantric_pos_idx = self._simi_ir(s_all, q_sslp, idx, spatial_pos_idx) # (batch_size, ndata), similarity
        # - hard mining
        # - hard mining
        s_whole = self._hard_mining(d_all, d_all_b, q_sslp, idx, spatial_pos_idx, semantric_pos_idx) #(bs , k=4096), similarity
        s_spatial_numerator = torch.sum((semantric_pos_idx.byte()|spatial_pos_idx.byte())*s_all, dim=1)
        loss_spatial = -torch.mean(torch.log(s_spatial_numerator/s_whole + 1e-7))

        neg_idx = self.memory_bank.multinomial.draw(bs_g * self.k)
        # neg_idx = neg_idx.view(bs, -1)
        # while True:
        #     wrong = (neg_idx == idx.view(-1, 1))
        #     if wrong.sum().item() > 0:
        #         neg_idx[wrong] = self.memory_bank.multinomial.draw(
        #             wrong.sum().item())
        #     else:
        #         break
        # neg_idx = neg_idx.flatten()
        pos_feat = torch.index_select(self.memory_bank.feature_bank, 0,
                                      idx)  # BXC
        neg_feat = torch.index_select(self.memory_bank.feature_bank, 0,
                                      neg_idx).view(bs_g, self.k,
                                                    feat_dim)  # BxKxC
        # neg_feat = torch.cat([neg_feat for _ in range(sum(self.num_crops))])
        # pos_logits = torch.einsum('nc,nc->n',
        #                           [pos_feat, q]).unsqueeze(-1)
        # neg_logits = torch.bmm(neg_feat, q.unsqueeze(2)).squeeze(2)

        # loss_npid = self.head(pos_logits, neg_logits)['loss_contra']
        pos_logits_g = torch.einsum('nc,nc->n',
                                  [pos_feat, q_sslp]).unsqueeze(-1)
        neg_logits_g = torch.bmm(neg_feat,  q_sslp.unsqueeze(2)).squeeze(2)

        pos_feat_tl = torch.index_select(self.memory_bank_tl.feature_bank, 0,
                                    idx)  # BXC
        pos_feat_tr = torch.index_select(self.memory_bank_tr.feature_bank, 0,
                                    idx)  # BXC
        pos_feat_bl = torch.index_select(self.memory_bank_bl.feature_bank, 0,
                                    idx)  # BXC
        pos_feat_br = torch.index_select(self.memory_bank_br.feature_bank, 0,
                                    idx)  # BXC
        pos_feat_l = torch.cat([pos_feat_tl, pos_feat_tr, pos_feat_bl, pos_feat_br])
        neg_feat_tl = torch.index_select(self.memory_bank_tl.feature_bank, 0,
                                      neg_idx).view(bs_l, int(self.k/self.num_crops[1]),
                                                    feat_dim)  # 4BxK/4xC
        neg_feat_tr = torch.index_select(self.memory_bank_tr.feature_bank, 0,
                                      neg_idx).view(bs_l, int(self.k/self.num_crops[1]),
                                                    feat_dim)  
        neg_feat_bl = torch.index_select(self.memory_bank_bl.feature_bank, 0,
                                      neg_idx).view(bs_l, int(self.k/self.num_crops[1]),
                                                    feat_dim)  
        neg_feat_br = torch.index_select(self.memory_bank_br.feature_bank, 0,
                                      neg_idx).view(bs_l, int(self.k/self.num_crops[1]),
                                                    feat_dim)  
        neg_feat_l = torch.cat([neg_feat_tl, neg_feat_tr, neg_feat_bl, neg_feat_br],1) # 4BxKxC
        pos_logits_l = torch.einsum('nc,nc->n',
                                  [pos_feat_l, q_l]).unsqueeze(-1)
        neg_logits_l = torch.bmm(neg_feat_l, q_l.unsqueeze(2)).squeeze(2)

        loss_npid = self.head(pos_logits_g, neg_logits_g)['loss_contra']
        loss_mcrop = self.head(pos_logits_l, neg_logits_l)['loss_contra']
        # w1 = self.loss_lambda1/(self.loss_lambda1+self.loss_lambda2 * self.warm_up)
        # w2 = 1-w1

        # sementic_weight 
        current = np.clip(kwargs['epoch'], 0.0, self.rampup_length)
        phase = 1.0 - current / self.rampup_length
        semnetic_weight = float(np.exp(-5.0 * phase * phase))
        # npid_weight = 1/(1 + semnetic_weight + semnetic_weight)
        # semnetic_weight = semnetic_weight/(1 + semnetic_weight + semnetic_weight)
        if kwargs['iter']==0 and torch.distributed.get_rank() == 0:
            print('epoch:{}, semantic weight{:.3f}:'.format(kwargs['epoch'], semnetic_weight))
        # gather all losses
        losses = dict()
        # losses['loss_contra_single'] = loss_npid * npid_weight
        # losses['loss_contra_spatial'] = loss_spatial * semnetic_weight 
        # losses['loss_contra_sementic'] = loss_semantic  * semnetic_weight
        losses['loss_contra_single'] = loss_npid * self.loss_lambda
        losses['loss_contra_spatial'] = loss_spatial * semnetic_weight * self.loss_lambda
        losses['loss_contra_mcrop'] = loss_mcrop * self.loss_lambda
        # losses['loss_contra_sementic'] = loss_semantic  * semnetic_weight 
            
        with torch.no_grad():
            # renew self.cluster
            # print('i=',i, 'rank=',torch.distributed.get_rank(),self.similar)
            if kwargs['iter']==0 and torch.distributed.get_rank() == 0 and self.similar:
                print('Fitting K-means with FAISS')
                km = Kmeans(self.kmeans, self.memory_bank.feature_bank, [0,1,2,3])
                cluster_labels = km.compute_clusters()
                self._cluster_update(cluster_labels)
            # update memory bank
            self.memory_bank.update(idx, q_sslp.detach())
            self.memory_bank_b.update(idx, x.detach())
            self.memory_bank_tl.update(idx, q_l[:bs_g].detach())
            self.memory_bank_tr.update(idx, q_l[bs_g:2*bs_g].detach())
            self.memory_bank_bl.update(idx, q_l[2*bs_g:3*bs_g].detach())
            self.memory_bank_br.update(idx, q_l[3*bs_g:].detach())
        return losses







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