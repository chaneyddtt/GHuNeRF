import torch.nn as nn
import torch.nn.functional as F
import torch

from spconv.pytorch.conv import (SparseConv2d, SparseConv3d,
                                 SparseConvTranspose2d,
                                 SparseConvTranspose3d, SparseInverseConv2d,
                                 SparseInverseConv3d, SubMConv2d, SubMConv3d)
from spconv.pytorch.core import SparseConvTensor
from spconv.pytorch.identity import Identity
from spconv.pytorch.modules import SparseModule, SparseSequential
from spconv.pytorch.ops import ConvAlgo
from spconv.pytorch.pool import SparseMaxPool2d, SparseMaxPool3d
from spconv.pytorch.tables import AddTable, ConcatTable

from lib.config import cfg
from lib.networks.encoder import SpatialEncoder
from lib.networks.unet import build_encoder
import math
import time

class SpatialKeyValue(nn.Module):

    def __init__(self):
        super(SpatialKeyValue, self).__init__()

        self.key_embed = nn.Conv1d(256, 128, kernel_size=1, stride=1)
        self.value_embed = nn.Conv1d(256, 256, kernel_size=1, stride=1)

    def forward(self, x):

        return (self.key_embed(x),
                self.value_embed(x))


class TemporalKeyValue(nn.Module):

    def __init__(self):
        super(TemporalKeyValue, self).__init__()

        self.key_embed = nn.Conv1d(256, 128, kernel_size=1, stride=1)
        self.value_embed = nn.Conv1d(256, 256, kernel_size=1, stride=1)
        self.query_embed = nn.Conv1d(256, 128, kernel_size=1, stride=1)

    def forward(self, x):

        return (self.key_embed(x),
                self.value_embed(x),
                self.query_embed(x))


class TemporalCrossTransformer(nn.Module):
    def __init__(self):
        super(TemporalCrossTransformer, self).__init__()
        self.temporal_key_value_0 = TemporalKeyValue()

    def forward(self, feat):
        key_embed, value_embed, query_embed = self.temporal_key_value_0(
            feat.permute(2, 1, 0))

        k_emb = key_embed.size(1)
        A = torch.bmm(query_embed.transpose(1, 2), key_embed)
        A = A / math.sqrt(k_emb)
        A = F.softmax(A, dim=1)
        out = torch.bmm(value_embed, A)

        final_holder = value_embed.permute(2, 1, 0) + out.permute(2, 1, 0)

        return final_holder

class BWRefine(nn.Module):
    def __init__(self):
        super(BWRefine, self).__init__()
        input_ch = 2*cfg.dist_res+1+24+69 if cfg.pose_encode else 2*cfg.dist_res+1+24
        D = 8
        W = 256
        self.skips = [4]
        self.bw_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
            self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
        ])
        self.bw_fc = nn.Conv1d(W, 24, 1)
        self.actvn = nn.ReLU()

    def forward(self, bw_init, dist, poses=None):
        features = torch.cat([bw_init, dist, poses], dim=1) if cfg.pose_encode else torch.cat([bw_init, dist], dim=1)
        net = features
        for i, l in enumerate(self.bw_linears):
            net = self.actvn(self.bw_linears[i](net))
            if i in self.skips:
                net = torch.cat([features, net], dim=1)
        bw = self.bw_fc(net)
        bw = torch.log(bw_init + 1e-9) + bw
        bw = F.softmax(bw, dim=1)
        return bw


def combine_interleaved(t, num_input=4, agg_type="average"):

    t = t.reshape(-1, num_input, *t.shape[1:])

    if agg_type == "average":
        t = torch.mean(t, dim=1)

    elif agg_type == "max":
        t = torch.max(t, dim=1)[0]
    else:
        raise NotImplementedError("Unsupported combine type " + agg_type)
    return t


def repeat_interleave(input, repeats, dim=0):
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])


def fused_mean_variance(x):
    mean = x.mean(0).unsqueeze(0)
    var = torch.mean((x - mean)**2, dim=0, keepdim=True)
    return mean, var


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.encoder = SpatialEncoder()
        if cfg.bw_refine:
            print("Refine blending weights")
            self.bw_refine = BWRefine()

        if cfg.weight == 'cross_transformer':
            self.spatial_key_value_0 = SpatialKeyValue()
            self.spatial_key_value_1 = SpatialKeyValue()

            self.spatial_key_value_2 = SpatialKeyValue()
            self.spatial_key_value_3 = SpatialKeyValue()

        self.temporal_att = TemporalCrossTransformer()

        self.xyzc_net = SparseConvNet()

        self.actvn = nn.ReLU()

        self.fc_0 = nn.Conv1d(384, 256, 1)
        self.fc_1 = nn.Conv1d(256, 256, 1)
        self.fc_2 = nn.Conv1d(256, 256, 1)
        self.alpha_fc = nn.Conv1d(256, 1, 1)

        self.feature_fc = nn.Conv1d(256, 256, 1)

        self.view_fc = nn.Conv1d(283, 128, 1)
        self.rgb_fc = nn.Conv1d(128, 3, 1)

        self.fc_3 = nn.Conv1d(256, 256, 1)
        self.fc_4 = nn.Conv1d(128, 128, 1)

        self.alpha_res_0 = nn.Conv1d(cfg.img_feat_size, 256, 1)

        self.rgb_res_0 = nn.Conv1d(cfg.img_feat_size, 256, 1)

    def cross_attention_density(self, holder, pixel_feat):

        key_embed, value_embed = self.spatial_key_value_0(
            pixel_feat.permute(2, 1, 0))

        query_key, query_value = self.spatial_key_value_1(
            holder.permute(2, 1, 0))
        k_emb = key_embed.size(1)
        A = torch.bmm(key_embed.transpose(1, 2), query_key)
        A = A / math.sqrt(k_emb)
        A = F.softmax(A, dim=1)
        out = torch.bmm(value_embed, A)

        final_holder = query_value.permute(2, 1, 0) + out.permute(2, 1, 0)

        return final_holder

    def cross_attention_color(self, holder, pixel_feat):

        key_embed, value_embed = self.spatial_key_value_2(
            pixel_feat.permute(2, 1, 0))

        query_key, query_value = self.spatial_key_value_3(
            holder.permute(2, 1, 0))
        k_emb = key_embed.size(1)
        A = torch.bmm(key_embed.transpose(1, 2), query_key)
        A = A / math.sqrt(k_emb)
        A = F.softmax(A, dim=1)
        out = torch.bmm(value_embed, A)

        final_holder = query_value.permute(2, 1, 0) + out.permute(2, 1, 0)

        return final_holder

    def forward(self, pixel_feat, sp_input, grid_coords, viewdir, light_pts,
                holder=None, cam_id=None):

        feature = sp_input['feature']
        coord = sp_input['coord']
        out_sh = sp_input['out_sh']
        batch_size = sp_input['batch_size']

        p_features = grid_coords.transpose(1, 2)
        grid_coords = grid_coords[:, None, None]

        xyz = feature[..., :3]

        B = light_pts.shape[0]
        # n_input = int(pixel_feat.shape[0] / B)
        n_input = holder.shape[0]
        xyzc_features_list = []

        for view in range(n_input):
            xyzc = SparseConvTensor(holder[view], coord, out_sh, batch_size)
            xyzc_feature = self.xyzc_net(xyzc, grid_coords)
            xyzc_features_list.append(xyzc_feature)

        xyzc_features = torch.cat(xyzc_features_list,dim=0)

        net = self.actvn(self.fc_0(xyzc_features))
        net = self.cross_attention_density(net,
                                   self.actvn(self.alpha_res_0(pixel_feat)))
        net = self.actvn(self.fc_1(net))

        inter_net = self.actvn(self.fc_2(net))

        opa_net = self.actvn(self.fc_3(inter_net))
        alpha = self.alpha_fc(opa_net)

        features = self.actvn(self.feature_fc(inter_net))

        features = self.cross_attention_color(features, self.actvn(self.rgb_res_0(pixel_feat)))

        viewdir = repeat_interleave(viewdir,n_input)
        viewdir = viewdir.transpose(1, 2)

        features = torch.cat((features, viewdir), dim=1)
        net = self.actvn(self.view_fc(features))

        net = self.actvn(self.fc_4(net))
        rgb = self.rgb_fc(net)

        raw = torch.cat((rgb, alpha), dim=1)
        raw = raw.transpose(1, 2)

        return raw


class SparseConvNet(nn.Module):
    def __init__(self):
        super(SparseConvNet, self).__init__()

        self.conv0 = double_conv(64, 64, 'subm0')
        self.down0 = stride_conv(64, 64, 'down0')

        self.conv1 = double_conv(64, 64, 'subm1')
        self.down1 = stride_conv(64, 64, 'down1')

        self.conv2 = triple_conv(64, 64, 'subm2')
        self.down2 = stride_conv(64, 128, 'down2')

        self.conv3 = triple_conv(128, 128, 'subm3')
        self.down3 = stride_conv(128, 128, 'down3')

        self.conv4 = triple_conv(128, 128, 'subm4')

    def forward(self, x, grid_coords):

        net = self.conv0(x)
        net = self.down0(net)

        net = self.conv1(net)
        net1 = net.dense()
        feature_1 = F.grid_sample(net1,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)

        net = self.down1(net)

        net = self.conv2(net)
        net2 = net.dense()
        feature_2 = F.grid_sample(net2,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        net = self.down2(net)

        net = self.conv3(net)
        net3 = net.dense()
        feature_3 = F.grid_sample(net3,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        net = self.down3(net)

        net = self.conv4(net)
        net4 = net.dense()
        feature_4 = F.grid_sample(net4,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        '''

        '''

        features = torch.cat((feature_1, feature_2, feature_3, feature_4),
                             dim=1)
        features = features.view(features.size(0), -1, features.size(4))

        return features


def single_conv(in_channels, out_channels, indice_key=None):
    return SparseSequential(
        SubMConv3d(in_channels,
                   out_channels,
                   1,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def double_conv(in_channels, out_channels, indice_key=None):
    return SparseSequential(
        SubMConv3d(in_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        SubMConv3d(out_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def triple_conv(in_channels, out_channels, indice_key=None):
    return SparseSequential(
        SubMConv3d(in_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        SubMConv3d(out_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        SubMConv3d(out_channels,
                   out_channels,
                   3,
                   bias=False,
                   indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def stride_conv(in_channels, out_channels, indice_key=None):
    return SparseSequential(
        SparseConv3d(in_channels,
                     out_channels,
                     3,
                     2,
                     padding=1,
                     bias=False,
                     indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU())
