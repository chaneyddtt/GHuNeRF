import torch
import torch.nn as nn
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
import matplotlib.pyplot as plt
import numpy as np
import gc
import math
import time


class Renderer:
    def __init__(self, net):
        self.net = net

    def paint_neural_human(self, batch, t, holder_feat_map, holder_feat_scale,
                           prev_weight=None, prev_holder=None):

        smpl_vertice = batch['smpl_vertice'][t]

        if cfg.rasterize:
            vizmap = batch['input_vizmaps'][t]

        image_shape = batch['input_imgs'][t].shape[-2:]

        input_R = batch['input_R']
        input_T = batch['input_T']
        input_K = batch['input_K']

        input_R = input_R.reshape(-1, 3, 3)
        input_T = input_T.reshape(-1, 3, 1)
        input_K = input_K.reshape(-1, 3, 3)

        if cfg.rasterize:
            result = vizmap[0]

        # uv
        vertice_rot = \
            torch.matmul(input_R[:, None], smpl_vertice.unsqueeze(-1))[..., 0]
        vertice = vertice_rot + input_T[:, None, :3, 0]
        vertice = torch.matmul(input_K[:, None], vertice.unsqueeze(-1))[..., 0]
        uv = vertice[:, :, :2] / vertice[:, :, 2:]


        latent = self.sample_from_feature_map(holder_feat_map,
                                              holder_feat_scale, image_shape,
                                              uv)

        latent = latent.permute(0, 2, 1)

        num_input = latent.shape[0]

        if cfg.use_viz_test:

            final_result = result
            big_holder = torch.zeros((latent.shape[0], latent.shape[1],
                                      cfg.embed_size)).cuda()
            big_holder[final_result == True, :] = latent[final_result == True,
                                                  :]

            if cfg.weight == 'cross_transformer':
                return final_result, big_holder

        else:  # not using viz test

            holder = latent.sum(0)
            holder = holder / num_input
            return holder

    def sample_from_feature_map(self, feat_map, feat_scale, image_shape, uv):

        scale = feat_scale / image_shape
        scale = torch.tensor(scale).to(dtype=torch.float32).to(
            device=torch.cuda.current_device())

        uv = uv * scale - 1.0
        uv = uv.unsqueeze(2)

        samples = F.grid_sample(
            feat_map,
            uv,
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        )

        return samples[:, :, :, 0]

    def get_pixel_aligned_feature(self, batch, xyz, pixel_feat_map,
                                  pixel_feat_scale, batchify=False):



        image_shape = batch['input_imgs'][0].shape[-2:]
        input_R = batch['input_R']
        input_T = batch['input_T']
        input_K = batch['input_K']

        input_R = input_R.reshape(-1, 3, 3)
        input_T = input_T.reshape(-1, 3, 1)
        input_K = input_K.reshape(-1, 3, 3)


        if batchify == False:
            xyz = xyz.view(xyz.shape[0], -1, 3)
        xyz = repeat_interleave(xyz, input_R.shape[0])

        xyz_rot = torch.matmul(input_R[:, None], xyz.unsqueeze(-1))[..., 0]
        xyz = xyz_rot + input_T[:, None, :3, 0]
        xyz = torch.matmul(input_K[:, None], xyz.unsqueeze(-1))[..., 0]
        uv = xyz[:, :, :2] / xyz[:, :, 2:]

        pixel_feat = self.sample_from_feature_map(pixel_feat_map,
                                                  pixel_feat_scale, image_shape,
                                                  uv)

        return pixel_feat

    def get_sampling_points(self, ray_o, ray_d, near, far):
        # calculate the steps for each ray
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

        if cfg.perturb > 0. and self.net.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper)
            z_vals = lower + (upper - lower) * t_rand

        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

        return pts, z_vals

    def pts_to_can_pts(self, pts, batch):
        """transform pts from the world coordinate to the smpl coordinate"""
        Th = batch['Th'][:, None]
        pts = pts - Th
        R = batch['R']
        sh = pts.shape
        pts = torch.matmul(pts.view(sh[0], -1, sh[3]), R)
        pts = pts.view(*sh)
        return pts

    def transform_sampling_points(self, pts, batch):
        if not self.net.training:
            return pts
        center = batch['center'][:, None, None]
        pts = pts - center
        rot = batch['rot']
        pts_ = pts[..., [0, 2]].clone()
        sh = pts_.shape
        pts_ = torch.matmul(pts_.view(sh[0], -1, sh[3]), rot.permute(0, 2, 1))
        pts[..., [0, 2]] = pts_.view(*sh)
        pts = pts + center
        trans = batch['trans'][:, None, None]
        pts = pts + trans
        return pts

    def prepare_sp_input(self, batch):
        sp_input = {}

        # feature: [N, f_channels]
        sh = batch['feature'].shape
        sp_input['feature'] = batch['feature'].view(-1, sh[-1])

        # coordinate: [N, 4], batch_idx, z, y, x
        sh = batch['coord'].shape
        idx = [torch.full([sh[1]], i) for i in range(sh[0])]
        idx = torch.cat(idx).to(batch['coord'])
        coord = batch['coord'].view(-1, sh[-1])
        sp_input['coord'] = torch.cat([idx[:, None], coord], dim=1)

        out_sh, _ = torch.max(batch['out_sh'], dim=0)
        sp_input['out_sh'] = out_sh.tolist()
        sp_input['batch_size'] = sh[0]

        sp_input['i'] = batch['i']

        return sp_input

    def get_grid_coords(self, pts, sp_input, batch):
        # convert xyz to the voxel coordinate dhw
        dhw = pts[..., [2, 1, 0]]
        min_dhw = batch['bounds'][:, 0, [2, 1, 0]]
        dhw = dhw - min_dhw[:, None]
        dhw = dhw / torch.tensor(cfg.voxel_size).to(dhw)
        # convert the voxel coordinate to [-1, 1]
        out_sh = torch.tensor(sp_input['out_sh']).to(dhw)
        dhw = dhw / out_sh * 2 - 1
        # convert dhw to whd, since the occupancy is indexed by dhw
        grid_coords = dhw[..., [2, 1, 0]]
        return grid_coords

    def prepare_inside_pts(self, pts, batch):
        if 'Ks' not in batch:
            __import__('ipdb').set_trace()
            return raw

        sh = pts.shape
        pts = pts.view(sh[0], -1, sh[3])

        insides = []

        for nv in range(batch['Ks'].size(1)):
            # project pts to image space
            R = batch['RT'][:, nv, :3, :3]
            T = batch['RT'][:, nv, :3, 3]
            pts_ = torch.matmul(pts, R.transpose(2, 1)) + T[:, None]
            pts_ = torch.matmul(pts_, batch['Ks'][:, nv].transpose(2, 1))
            pts2d = pts_[..., :2] / pts_[..., 2:]

            # ensure that pts2d is inside the image
            pts2d = pts2d.round().long()
            H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
            pts2d[..., 0] = torch.clamp(pts2d[..., 0], 0, W - 1)
            pts2d[..., 1] = torch.clamp(pts2d[..., 1], 0, H - 1)

            # remove the points outside the mask
            pts2d = pts2d[0]
            msk = batch['msks'][0, nv]

            inside = msk[pts2d[:, 1], pts2d[:, 0]][None].bool()

            insides.append(inside)

        inside = insides[0]
        for i in range(1, len(insides)):
            inside = inside * insides[i]


        return inside

    def batchify_rays(self,
                      sp_input,
                      grid_coords,
                      viewdir,
                      light_pts,
                      chunk=1024 * 32,
                      net_c=None,
                      batch=None,
                      xyz=None,
                      pixel_feat_map=None,
                      pixel_feat_scale=None,
                      norm_viewdir=None,
                      holder=None,
                      embed_xyz=None):
        """Render rays in smaller minibatches to avoid OOM.
        """

        all_ret = []

        for i in range(0, grid_coords.shape[1], chunk):

            xyz_shape = xyz.shape
            xyz = xyz.reshape(xyz_shape[0], -1, 3)

            pixel_feat = self.get_pixel_aligned_feature(batch,
                                                        xyz[:, i:i + chunk],
                                                        pixel_feat_map,
                                                        pixel_feat_scale,
                                                        batchify=True)

            ret = self.net(pixel_feat, sp_input,
                           grid_coords[:, i:i + chunk],
                           viewdir[:, i:i + chunk],
                           light_pts[:, i:i + chunk],
                           holder=holder)

            all_ret.append(ret)


        all_ret = torch.cat(all_ret, 1)

        return all_ret


    def render(self, batch):
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']
        bw = batch['pbw']
        poses = batch['pose_target'][:, :, None]
        sh = ray_o.shape

        pts, z_vals = self.get_sampling_points(ray_o, ray_d, near, far)

        xyz = pts.clone()
        light_pts = embedder.xyz_embedder(pts)

        pts = self.pts_to_can_pts(pts, batch)
        pbw = bounds_grid_sample_blend_weights(pts.view(pts.shape[0], -1, 3), bw, batch['bounds'])
        n_batch = pbw.shape[0]
        n_point = pbw.shape[-1]
        pnorm = pbw[:, -1]
        pind = pnorm < cfg.norm_th
        pind[torch.arange(len(pnorm)), pnorm.argmin(dim=1)] = True
        pbw = pbw[:, :-1, pind[0]]
        pnorm = pnorm[:, pind[0]]
        pind = pind.view(pind.shape[0], pts.shape[1], pts.shape[2])
        pts = pts[pind][None]

        if cfg.bw_refine:
            dist_embed = embedder.dist_embedder(pnorm[:, None].transpose(1, 2))
            if cfg.pose_encode:
                pbw = self.net.bw_refine(pbw, dist_embed.transpose(1, 2), poses.expand(-1, -1, pbw.shape[-1]))
            else:
                pbw = self.net.bw_refine(pbw, dist_embed.transpose(1, 2))

        ray_d0 = batch['ray_d']

        viewdir = ray_d0 / torch.norm(ray_d0, dim=2, keepdim=True)
        viewdir = embedder.view_embedder(viewdir)
        viewdir = viewdir[:, :, None].repeat(1, 1, xyz.size(2), 1).contiguous()
        viewdir = viewdir[pind][None]

        xyz = xyz[pind][None]
        light_pts = light_pts[pind][None]

        light_pts = light_pts.view(sh[0], -1,
                                   embedder.xyz_dim)
        viewdir = viewdir.view(sh[0], -1,
                               embedder.view_dim)

        sp_input = self.prepare_sp_input(batch)
        grid_coords = self.get_grid_coords(pts, sp_input,
                                           batch)
        grid_coords = grid_coords.view(sh[0], -1, 3)

        if ray_o.size(1) <= 2048:
            xyz_transform = self.transform_pts2frames(pbw, xyz.view(xyz.shape[0], -1, 3), batch)
        else:
            xyz_transform = self.transform_pts2frames(pbw, xyz.view(xyz.shape[0], -1, 3), batch, batchify=True)


        image_list = batch['input_imgs']

        weight = None
        holder = None

        images = (torch.cat(image_list, dim=1)).squeeze(0)
        images_resize = transform(images)
        holder_feat_map, holder_feat_scale, pixel_feat_map, pixel_feat_scale = self.net.encoder(images)
        _, holder_img = self.paint_neural_human(batch, images_resize, holder_feat_scale, weight, holder,
                                                use_pixel=True)
        weight, holder = self.paint_neural_human(batch,
                                                 holder_feat_map,
                                                 holder_feat_scale,
                                                 weight, holder)

        if cfg.use_viz_test:
            holder = torch.sum(holder * weight[..., None], dim=0, keepdim=True) / (
                        torch.sum(weight[..., None], dim=0, keepdim=True) + 1e-8)
            holder_img = torch.sum(holder_img * weight[..., None], dim=0) / (
                    torch.sum(weight[..., None], dim=0) + 1e-8)
        else:
            holder = torch.mean(holder, dim=0, keepdim=True)
            holder_img = torch.mean(holder_img, dim=0, keepdim=True)

        if ray_o.size(1) <= 2048:

            pixel_feat = self.get_pixel_aligned_feature(batch, xyz_transform,
                                                        pixel_feat_map,
                                                        pixel_feat_scale)

            if cfg.xyz_refine:
                xyz_init = xyz_transform.clone().detach().permute(0, 2, 1)
                dxyz = self.net.xyz_refine(xyz_init, pixel_feat.clone().detach())
                xyz_transform_re = xyz_transform + dxyz
                pixel_feat = self.get_pixel_aligned_feature(batch, xyz_transform_re, pixel_feat_map, pixel_feat_scale)

            # for debugging
            # pixel_feat_image = self.get_pixel_aligned_feature(batch, xyz_transform, images_resize, pixel_feat_scale)
            # pixel_feat_image = torch.mean(pixel_feat_image.permute(0, 2, 1), dim=0)
            # pixel_feat_image = pixel_feat_image.detach()

            raw = self.net(pixel_feat, sp_input, grid_coords, viewdir,
                           light_pts, holder=holder, cam_id=batch['cam_embid'])  # , dist=smpl_dist)
            # image_rgb=pixel_feat_image)

        else:
            # for debugging
            # pixel_feat_image = self.get_pixel_aligned_feature(batch, xyz_transform, images_resize, pixel_feat_scale)
            # pixel_feat_image = torch.mean(pixel_feat_image.permute(0, 2, 1), dim=0)

            raw = self.batchify_rays(sp_input, grid_coords, viewdir,
                                     light_pts,
                                     chunk=1024 * 32, net_c=None,
                                     batch=batch, xyz=xyz_transform,
                                     pixel_feat_map=pixel_feat_map,
                                     pixel_feat_scale=pixel_feat_scale,
                                     holder=holder, cam_id=batch['cam_embid'])  # smpl_dist=smpl_dist)
            # image_rgb=pixel_feat_image)
            dxyz = torch.zeros_like(xyz_transform)

        raw_full = torch.zeros([n_batch, n_point, 4], dtype=pts.dtype, device=pts.device)
        raw_full[pind.view(pind.shape[0], -1)] = raw
        raw = raw_full.reshape(-1, z_vals.size(2), 4)
        z_vals = z_vals.view(-1, z_vals.size(2))
        ray_d = ray_d.view(-1, 3)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, ray_d, cfg.raw_noise_std, cfg.white_bkgd)

        rgb_map = rgb_map.view(*sh[:-1], -1)
        acc_map = acc_map.view(*sh[:-1])
        depth_map = depth_map.view(*sh[:-1])

        ret = {'rgb_map': rgb_map, 'acc_map': acc_map, 'depth_map': depth_map}

        if cfg.run_mode == 'test':
            gc.collect()
            torch.cuda.empty_cache()
            print(torch.cuda.memory_allocated())

        return ret

