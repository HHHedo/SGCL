import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS
import numpy as np

@MODELS.register_module
class PIRL(nn.Module):
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
                 num_crops=[1, 3],
                 k=4096,
                 **kwargs):
        super(PIRL, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)
        self.memory_bank = builder.build_memory(memory_bank)
        self.init_weights(pretrained=pretrained)
        self.loss_lambda = loss_lambda
        self.num_crops = num_crops
        self.k=k
      

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights(init_linear='kaiming')


    def forward_train(self, img, img2, repeat_idx, idx, bag_idx, x_coord, y_coord, **kwargs):
        assert isinstance(img2, list)
        img=img2 #list of tensors, each tensor[bs,3,224]
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
        q, q_l = self.neck(feature)
        q_sslp = nn.functional.normalize(q)  # BxC
        q_l = nn.functional.normalize(q_l)  # BxC
        bs_g, feat_dim = q_sslp.shape[:2]
        bs_l, _ = q_l.shape[:2]
        
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


        pos_logits_jig = torch.einsum('nc,nc->n',
                                  [pos_feat, q_l]).unsqueeze(-1)
        neg_logits_jig = torch.bmm(neg_feat, q_l.unsqueeze(2)).squeeze(2)




        loss_npid = self.head(pos_logits_g, neg_logits_g)['loss_contra']
        loss_jig = self.head(pos_logits_jig, neg_logits_jig)['loss_contra']

       
        losses = dict()
        # losses['loss_contra_single'] = loss_npid * npid_weight
        # losses['loss_contra_spatial'] = loss_spatial * semnetic_weight 
        # losses['loss_contra_sementic'] = loss_semantic  * semnetic_weight
        losses['loss_contra_single'] = loss_npid *0.5
        losses['loss_contra_jig'] = loss_jig *0.5

        # losses['loss_contra_sementic'] = loss_semantic  * semnetic_weight 
            
        with torch.no_grad():
            if kwargs['iter']%50==0 and torch.distributed.get_rank() == 0:
                g_v_p, g_m_p = torch.var_mean(torch.exp(pos_logits_g/self.T))
                g_v_n, g_m_n  = torch.var_mean(torch.exp(neg_logits_g/self.T))
                jig_v_p, jig_m_p  = torch.var_mean(torch.exp(pos_logits_jig/self.T))
                jig_v_n, jig_m_n = torch.var_mean(torch.exp(neg_logits_jig/self.T))
                print('IR: {:.3f}+-{:.3f}/{:.3f}+-{:.3f}, Jig: {:.3f}+-{:.3f}/{:.3f}+-{:.3f}'
                .format(g_m_p , g_v_p, g_m_n , g_v_n, jig_m_p ,jig_v_p, jig_m_n ,jig_v_n))
            self.memory_bank.update(idx, q_sslp.detach())
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





