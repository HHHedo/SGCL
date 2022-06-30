import torch
import torch.nn as nn
from packaging import version
from mmcv.cnn import kaiming_init, normal_init

from .registry import NECKS
from .utils import build_norm_layer


def _init_weights(module, init_linear='normal', std=0.01, bias=0.):
    assert init_linear in ['normal', 'kaiming'], \
        "Undefined init_linear: {}".format(init_linear)
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if init_linear == 'normal':
                normal_init(m, std=std, bias=bias)
            else:
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                            nn.GroupNorm, nn.SyncBatchNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


@NECKS.register_module
class LinearNeck(nn.Module):
    """Linear neck: fc only.
    """

    def __init__(self, in_channels, out_channels, with_avg_pool=True):
        super(LinearNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, out_channels)

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.fc(x.view(x.size(0), -1))]


@NECKS.register_module
class RelativeLocNeck(nn.Module):
    """Relative patch location neck: fc-bn-relu-dropout.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 sync_bn=False,
                 with_avg_pool=True):
        super(RelativeLocNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if version.parse(torch.__version__) < version.parse("1.4.0"):
            self.expand_for_syncbn = True
        else:
            self.expand_for_syncbn = False

        self.fc = nn.Linear(in_channels * 2, out_channels)
        if sync_bn:
            _, self.bn = build_norm_layer(
                dict(type='SyncBN', momentum=0.003),
                out_channels)
        else:
            self.bn = nn.BatchNorm1d(
                out_channels, momentum=0.003)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()
        self.sync_bn = sync_bn

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear, std=0.005, bias=0.1)

    def _forward_syncbn(self, module, x):
        assert x.dim() == 2
        if self.expand_for_syncbn:
            x = module(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        else:
            x = module(x)
        return x

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.sync_bn:
            x = self._forward_syncbn(self.bn, x)
        else:
            x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return [x]


@NECKS.register_module
class NonLinearNeckV0(nn.Module):
    """The non-linear neck in ODC, fc-bn-relu-dropout-fc-relu.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 sync_bn=False,
                 with_avg_pool=True):
        super(NonLinearNeckV0, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if version.parse(torch.__version__) < version.parse("1.4.0"):
            self.expand_for_syncbn = True
        else:
            self.expand_for_syncbn = False

        self.fc0 = nn.Linear(in_channels, hid_channels)
        if sync_bn:
            _, self.bn0 = build_norm_layer(
                dict(type='SyncBN', momentum=0.001, affine=False),
                hid_channels)
        else:
            self.bn0 = nn.BatchNorm1d(
                hid_channels, momentum=0.001, affine=False)

        self.fc1 = nn.Linear(hid_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()
        self.sync_bn = sync_bn

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def _forward_syncbn(self, module, x):
        assert x.dim() == 2
        if self.expand_for_syncbn:
            x = module(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        else:
            x = module(x)
        return x

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        if self.sync_bn:
            x = self._forward_syncbn(self.bn0, x)
        else:
            x = self.bn0(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.relu(x)
        return [x]


@NECKS.register_module
class NonLinearNeckV1(nn.Module):
    """The non-linear neck in MoCo v2: fc-relu-fc.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True):
        super(NonLinearNeckV1, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.mlp(x.view(x.size(0), -1))]

@NECKS.register_module
class NonLinearNeckV11(nn.Module):
    """The non-linear neck in MoCo v2: fc-relu-fc.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True):
        super(NonLinearNeckV11, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [x.view(x.size(0), -1), self.mlp(x.view(x.size(0), -1))]

@NECKS.register_module
class NonLinearNeckV2(nn.Module):
    """The non-linear neck in byol: fc-bn-relu-fc.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True):
        super(NonLinearNeckV2, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.BatchNorm1d(hid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1, "Got: {}".format(len(x))
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.mlp(x.view(x.size(0), -1))]


@NECKS.register_module
class NonLinearNeckSimCLR(nn.Module):
    """SimCLR non-linear neck.

    Structure: fc(no_bias)-bn(has_bias)-[relu-fc(no_bias)-bn(no_bias)].
        The substructures in [] can be repeated. For the SimCLR default setting,
        the repeat time is 1.
    However, PyTorch does not support to specify (weight=True, bias=False).
        It only support \"affine\" including the weight and bias. Hence, the
        second BatchNorm has bias in this implementation. This is different from
        the official implementation of SimCLR.
    Since SyncBatchNorm in pytorch<1.4.0 does not support 2D input, the input is
        expanded to 4D with shape: (N,C,1,1). Not sure if this workaround
        has no bugs. See the pull request here:
        https://github.com/pytorch/pytorch/pull/29626.

    Args:
        num_layers (int): Number of fc layers, it is 2 in the SimCLR default setting.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_layers=2,
                 sync_bn=True,
                 with_bias=False,
                 with_last_bn=True,
                 with_avg_pool=True):
        super(NonLinearNeckSimCLR, self).__init__()
        self.sync_bn = sync_bn
        self.with_last_bn = with_last_bn
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if version.parse(torch.__version__) < version.parse("1.4.0"):
            self.expand_for_syncbn = True
        else:
            self.expand_for_syncbn = False

        self.relu = nn.ReLU(inplace=True)
        self.fc0 = nn.Linear(in_channels, hid_channels, bias=with_bias)
        if sync_bn:
            _, self.bn0 = build_norm_layer(
                dict(type='SyncBN'), hid_channels)
        else:
            self.bn0 = nn.BatchNorm1d(hid_channels)

        self.fc_names = []
        self.bn_names = []
        for i in range(1, num_layers):
            this_channels = out_channels if i == num_layers - 1 \
                else hid_channels
            self.add_module(
                "fc{}".format(i),
                nn.Linear(hid_channels, this_channels, bias=with_bias))
            self.fc_names.append("fc{}".format(i))
            if i != num_layers - 1 or self.with_last_bn:
                if sync_bn:
                    self.add_module(
                        "bn{}".format(i),
                        build_norm_layer(dict(type='SyncBN'), this_channels)[1])
                else:
                    self.add_module(
                        "bn{}".format(i),
                        nn.BatchNorm1d(this_channels))
                self.bn_names.append("bn{}".format(i))
            else:
                self.bn_names.append(None)

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def _forward_syncbn(self, module, x):
        assert x.dim() == 2
        if self.expand_for_syncbn:
            x = module(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        else:
            x = module(x)
        return x

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        if self.sync_bn:
            x = self._forward_syncbn(self.bn0, x)
        else:
            x = self.bn0(x)
        for fc_name, bn_name in zip(self.fc_names, self.bn_names):
            fc = getattr(self, fc_name)
            x = self.relu(x)
            x = fc(x)
            if bn_name is not None:
                bn = getattr(self, bn_name)
                if self.sync_bn:
                    x = self._forward_syncbn(bn, x)
                else:
                    x = bn(x)
        return [x]


@NECKS.register_module
class AvgPoolNeck(nn.Module):
    """Average pooling neck.
    """

    def __init__(self):
        super(AvgPoolNeck, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self, **kwargs):
        pass

    def forward(self, x):
        assert len(x) == 1
        return [self.avg_pool(x[0])]


@NECKS.register_module
class DenseCLNeck(nn.Module):
    '''The non-linear neck in DenseCL.
        Single and dense in parallel: fc-relu-fc, conv-relu-conv
    '''
    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_grid=None):
        super(DenseCLNeck, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

        self.with_pool = num_grid != None
        if self.with_pool:
            self.pool = nn.AdaptiveAvgPool2d((num_grid, num_grid))
        self.mlp2 = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, out_channels, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]

        avgpooled_x = self.avgpool(x)
        avgpooled_x = self.mlp(avgpooled_x.view(avgpooled_x.size(0), -1))

        if self.with_pool:
            x = self.pool(x) # sxs
        x = self.mlp2(x) # sxs: bxdxsxs
        avgpooled_x2 = self.avgpool2(x) # 1x1: bxdx1x1
        x = x.view(x.size(0), x.size(1), -1) # bxdxs^2
        avgpooled_x2 = avgpooled_x2.view(avgpooled_x2.size(0), -1) # bxd
        return [avgpooled_x, x, avgpooled_x2]

@NECKS.register_module
class DenseNPIDCLNeck(nn.Module):
    '''The non-linear neck in DenseCL.
        Single and dense in parallel: fc-relu-fc, conv-relu-conv
    '''
    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_grid=None):
        super(DenseNPIDCLNeck, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

        self.with_pool = num_grid != None
        if self.with_pool:
            self.pool = nn.AdaptiveAvgPool2d((num_grid, num_grid))
        self.mlp2 = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, out_channels, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        #add NPID
        self.mlp3 = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]

        avgpooled_x = self.avgpool(x)
        #add NPID
        NPID_avgpooled_x = self.mlp3(avgpooled_x.view(avgpooled_x.size(0), -1))
        avgpooled_x = self.mlp(avgpooled_x.view(avgpooled_x.size(0), -1))

        if self.with_pool:
            x = self.pool(x) # sxs
        x = self.mlp2(x) # sxs: bxdxsxs
        avgpooled_x2 = self.avgpool2(x) # 1x1: bxdx1x1
        x = x.view(x.size(0), x.size(1), -1) # bxdxs^2
        avgpooled_x2 = avgpooled_x2.view(avgpooled_x2.size(0), -1) # bxd
        return [avgpooled_x, x, avgpooled_x2, NPID_avgpooled_x]


@NECKS.register_module
class DoubleNonLinearNeckV1(nn.Module):
    """The non-linear neck in MoCo v2: fc-relu-fc.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True):
        super(DoubleNonLinearNeckV1, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))
        self.mlp2 = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.mlp(x.view(x.size(0), -1)), self.mlp2(x.view(x.size(0), -1))]



@NECKS.register_module
class SwAVNeck(nn.Module):
    """The non-linear neck of SwAV without bn & normalization: fc-relu-fc.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        with_avg_pool (bool): Whether to apply the global average pooling after
            backbone. Defaults to True.
        with_l2norm (bool): whether to normalize the output after projection.
            Defaults to True.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='SyncBN').
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True,
                 with_l2norm=False,
                 ):
        super(SwAVNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.with_l2norm = with_l2norm
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if out_channels == 0:
            self.projection_neck = None
        elif hid_channels == 0:
            self.projection_neck = nn.Linear(in_channels, out_channels)
        else:
            # self.bn = build_norm_layer(norm_cfg, hid_channels)[1]
            self.projection_neck = nn.Sequential(
                nn.Linear(in_channels, hid_channels),
                nn.ReLU(inplace=True), nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)
        
    def forward_projection(self, x):
        if self.projection_neck is not None:
            x = self.projection_neck(x)
        if self.with_l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)
        return x

    def forward(self, x):
        # forward computing
        # x: list of feature maps, len(x) according to len(num_crops)
        avg_out = []
        for _x in x:
            _x = _x[0]
            if self.with_avg_pool:
                _out = self.avgpool(_x)
                avg_out.append(_out)
        feat_vec = torch.cat(avg_out)  # [sum(num_crops) * N, C]
        feat_vec = feat_vec.view(feat_vec.size(0), -1)
        output = self.forward_projection(feat_vec)
        return [output]


@NECKS.register_module
class DuSwAVNeck(nn.Module):
    """The non-linear neck of SwAV without bn & normalization: fc-relu-fc.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        with_avg_pool (bool): Whether to apply the global average pooling after
            backbone. Defaults to True.
        with_l2norm (bool): whether to normalize the output after projection.
            Defaults to True.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='SyncBN').
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True,
                 with_l2norm=False,
                 ):
        super(DuSwAVNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.with_l2norm = with_l2norm
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if out_channels == 0:
            self.projection_neck = None
        elif hid_channels == 0:
            self.projection_neck = nn.Linear(in_channels, out_channels)
        else:
            # self.bn = build_norm_layer(norm_cfg, hid_channels)[1]
            self.projection_neck1 = nn.Sequential(
                nn.Linear(in_channels, hid_channels),
                nn.ReLU(inplace=True), nn.Linear(hid_channels, out_channels))
            self.projection_neck2 = nn.Sequential(
                nn.Linear(in_channels, hid_channels),
                nn.ReLU(inplace=True), nn.Linear(hid_channels, out_channels))


    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)
        
    def forward_projection(self, x, head_num):
        if head_num == 1:
            x = self.projection_neck1(x)
        elif head_num == 2:
            x = self.projection_neck2(x)
        if self.with_l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)
        return x

    def forward(self, x):
        # forward computing
        # x: list of feature maps, len(x) according to len(num_crops)
        avg_out = []
        for _x in x:
            _x = _x[0]
            if self.with_avg_pool:
                _out = self.avgpool(_x)
                avg_out.append(_out)
        feat_vec = torch.cat(avg_out)  # [sum(num_crops) * N, C]
        feat_vec = feat_vec.view(feat_vec.size(0), -1)
        output1 = self.forward_projection(feat_vec, 1)
        output2 = self.forward_projection(feat_vec, 2)
        return [output1, output2]



@NECKS.register_module
class JigsawNeck(nn.Module):
    """The non-linear neck of jigsaw without bn & normalization: fc-relu-fc.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        with_avg_pool (bool): Whether to apply the global average pooling after
            backbone. Defaults to True.
        with_l2norm (bool): whether to normalize the output after projection.
            Defaults to True.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='SyncBN').
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True,
                 jig_num=9,
                 bs=32,
                 ):
        super(JigsawNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.jig_num = jig_num
        self.bs=32
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.projection_neck1 = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.ReLU(inplace=True), nn.Linear(hid_channels, out_channels))
        self.projection_neck2 = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.ReLU(inplace=True), nn.Linear(hid_channels, out_channels),
            nn.ReLU(inplace=True))
        self.projection_neck3 =  nn.Linear(out_channels*9, out_channels)
            


    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)
        
    def forward_projection(self, x, head_num):
        if head_num == 1:
            x = self.projection_neck1(x)
        elif head_num == 2:
            x = self.projection_neck2(x)
            x_chunk = torch.chunk(x, chunks=self.jig_num, dim=0)
            jigsaw_featues = torch.cat(x_chunk, dim=1)
            # jigsaw_featues=[]
            # for i in range(self.jig_num):
            #     jigsaw_featues.append(x[i*self.bs:(i+1)*self.bs])
            # jigsaw_featues = torch.cat(jigsaw_featues,1)
            x = self.projection_neck3(jigsaw_featues)

        return x

    def forward(self, x):
        # forward computing
        # x: list of feature maps, len(x) according to len(num_crops)
        global_f, local_f = x
        global_f =  self.avgpool(global_f[0])
        local_f =  self.avgpool(local_f[0])
        # avg_out = []
        # for _x in x:
        #     _x = _x[0]
        #     if self.with_avg_pool:
        #         _out = self.avgpool(_x)
        #         avg_out.append(_out)
        # feat_vec = torch.cat(avg_out)  # [sum(num_crops) * N, C]
        # feat_vec = feat_vec.view(feat_vec.size(0), -1)
        output1 = self.forward_projection(global_f.view(global_f.size(0),-1), 1)
        output2 = self.forward_projection(local_f.view(local_f.size(0),-1), 2)
        return [output1, output2]
