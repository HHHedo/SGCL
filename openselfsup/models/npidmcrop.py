import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS


@MODELS.register_module
class NPIDMcrop(nn.Module):
    """NPID.

    Implementation of "Unsupervised Feature Learning via Non-parametric
    Instance Discrimination (https://arxiv.org/abs/1805.01978)".

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        memory_bank (dict): Config dict for module of memory banks. Default: None.
        neg_num (int): Number of negative samples for each image. Default: 65536.
        ensure_neg (bool): If False, there is a small probability
            that negative samples contain positive ones. Default: False.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 memory_bank=None,
                #  memory_bank_b=None, 
                 memory_bank_tr=None, 
                 memory_bank_tl=None, 
                 memory_bank_br=None, 
                 memory_bank_bl=None, 
                 num_crops=[1, 3],
                 neg_num=65536,
                 ensure_neg=False,
                 pretrained=None):
        super(NPIDMcrop, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.memory_bank_tl = builder.build_memory(memory_bank_tl)
        self.memory_bank_tr = builder.build_memory(memory_bank_tr)
        self.memory_bank_bl = builder.build_memory(memory_bank_bl)
        self.memory_bank_br = builder.build_memory(memory_bank_br)
        # self.memory_bank_b = builder.build_memory(memory_bank_b)
        self.pool =  nn.AdaptiveAvgPool2d((1, 1))
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)
        # self.proj = nn.Linear(512, 128)
        self.memory_bank = builder.build_memory(memory_bank)
        self.init_weights(pretrained=pretrained)
        self.num_crops = num_crops
        self.neg_num = neg_num
        self.ensure_neg = ensure_neg

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights(init_linear='kaiming')

    def forward_backbone(self, img):
        """Forward backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(img)
        return x

    
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
        q_g = q_norm[:x.shape[0]]
        q_l = q_norm[x.shape[0]:]
        bs_g, feat_dim = q_g.shape[:2]
        bs_l, _ = q_l.shape[:2]
        neg_idx = self.memory_bank.multinomial.draw(bs_g * self.neg_num)
        # if self.ensure_neg:
        #     neg_idx = neg_idx.view(bs_g, -1)
        #     while True:
        #         wrong = (neg_idx == idx.view(-1, 1))
        #         if wrong.sum().item() > 0:
        #             neg_idx[wrong] = self.memory_bank.multinomial.draw(
        #                 wrong.sum().item())
        #         else:
        #             break
        #     neg_idx = neg_idx.flatten()

        pos_feat = torch.index_select(self.memory_bank.feature_bank, 0,
                                      idx)  # BXC
        neg_feat = torch.index_select(self.memory_bank.feature_bank, 0,
                                      neg_idx).view(bs_g, self.neg_num,
                                                    feat_dim)  # BxKxC

        pos_logits_g = torch.einsum('nc,nc->n',
                                  [pos_feat, q_g]).unsqueeze(-1)
        neg_logits_g = torch.bmm(neg_feat, q_g.unsqueeze(2)).squeeze(2)
        # pos_feat2 =  torch.cat([pos_feat for _ in range(self.num_crops[1])])
        # neg_feat2 =  torch.cat([neg_feat for _ in range(self.num_crops[1])])
        # pos_logits_g2l = torch.einsum('nc,nc->n',
        #                           [pos_feat2, q_l]).unsqueeze(-1)
        # neg_logits_g2l = torch.bmm(neg_feat2,  q_l.unsqueeze(2)).squeeze(2)
        # q_l2g = self.proj(torch.cat([q[bs_g:2*bs_g],  q[2*bs_g:3*bs_g], q[3*bs_g:4*bs_g], q[4*bs_g:]], 1)) #b*4c --b*c
        # q_l2g = nn.functional.normalize(q_l2g)
        # pos_logits_g2l = torch.einsum('nc,nc->n',
        #                           [pos_feat, q_l2g]).unsqueeze(-1)
        # neg_logits_g2l = torch.bmm(neg_feat,  q_l2g.unsqueeze(2)).squeeze(2)
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
                                      neg_idx).view(bs_l, int(self.neg_num/self.num_crops[1]),
                                                    feat_dim)  # 4BxK/4xC
        neg_feat_tr = torch.index_select(self.memory_bank_tr.feature_bank, 0,
                                      neg_idx).view(bs_l, int(self.neg_num/self.num_crops[1]),
                                                    feat_dim)  
        neg_feat_bl = torch.index_select(self.memory_bank_bl.feature_bank, 0,
                                      neg_idx).view(bs_l, int(self.neg_num/self.num_crops[1]),
                                                    feat_dim)  
        neg_feat_br = torch.index_select(self.memory_bank_br.feature_bank, 0,
                                      neg_idx).view(bs_l, int(self.neg_num/self.num_crops[1]),
                                                    feat_dim)  
        neg_feat_l = torch.cat([neg_feat_tl, neg_feat_tr, neg_feat_bl, neg_feat_br],1) # 4BxKxC
        pos_logits_l2l = torch.einsum('nc,nc->n',
                                  [pos_feat_l, q_l]).unsqueeze(-1)
        neg_logits_l2l = torch.bmm(neg_feat_l, q_l.unsqueeze(2)).squeeze(2)

        loss_npid = self.head(pos_logits_g, neg_logits_g)['loss_contra']
        # loss_npid2 = self.head(pos_logits_g2l, neg_logits_g2l)['loss_contra']
        loss_mcrop = self.head(pos_logits_l2l, neg_logits_l2l)['loss_contra']
        losses = dict()
        losses['loss_contra_single'] = loss_npid * 0.2
        losses['loss_contra_mcrop'] = loss_mcrop * 0.8
        # losses['loss_contra_single2'] = loss_npid2 * 0.5

        # update memory bank
        with torch.no_grad():
            if kwargs['iter']%50==0 and torch.distributed.get_rank() == 0:
                # sp_num = spatial_pos_idx.float().sum(1).mean()
                # sp_var, sp_m = torch.var_mean(torch.sum((spatial_pos_idx.byte())*s_all, dim=1)/spatial_pos_idx.float().sum(1))
                # se_num = semantric_pos_idx.float().sum(1).mean()
                # se_var, se_m = torch.var_mean(torch.sum((semantric_pos_idx.byte())*s_all, dim=1)/semantric_pos_idx.float().sum(1))
                # hard_neg =  ((s_whole - s_spatial_numerator)/self.nei_k).mean()
                g_v_p, g_m_p = torch.var_mean(torch.exp(pos_logits_g/0.07))
                g_v_n, g_m_n  = torch.var_mean(torch.exp(neg_logits_g/0.07))
                l_v_p, l_m_p = torch.var_mean(torch.exp(pos_logits_l2l/0.07))
                l_v_n, l_m_n  = torch.var_mean(torch.exp(neg_logits_l2l/0.07))
                print(' IR:{:.1e}+-{:.1e}/{:.1e}+-{:.1e}\
                     Mcrop:{:.1e}+-{:.1e}/{:.1e}+-{:.1e}\
                         IR:{:.1e}/{:.1e}\
                     Mcrop:{:.1e}/{:.1e}'
                .format(g_m_p , g_v_p, g_m_n , g_v_n,
                l_m_p , l_v_p, l_m_n , l_v_n,
                torch.exp(pos_logits_g/0.2).mean(),
                torch.exp(neg_logits_g/0.2).mean(),
                torch.exp(pos_logits_l2l/0.2).mean(),
                torch.exp(neg_logits_l2l/0.2).mean()))
            self.memory_bank.update(idx, q_g.detach())
            self.memory_bank_tl.update(idx, q_l[:bs_g].detach())
            self.memory_bank_tr.update(idx, q_l[bs_g:2*bs_g].detach())
            self.memory_bank_bl.update(idx, q_l[2*bs_g:3*bs_g].detach())
            self.memory_bank_br.update(idx, q_l[3*bs_g:].detach())

        return losses

    def forward_test(self, img, **kwargs):
        pass

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.forward_backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))
