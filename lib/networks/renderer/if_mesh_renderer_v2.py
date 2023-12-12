import torch
import torch.nn as nn
import torchvision.transforms.functional

from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
from lib.utils.blend_utils import pts_sample_blend_weights, bounds_grid_sample_blend_weights
from zju_smpl.smplmodel.lbs import batch_rodrigues
import matplotlib.pyplot as plt
import numpy as np
import gc
import math
import time
import torchvision.transforms as T
import open3d
import mcubes
import trimesh
cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
transform = T.Resize(size=(256, 256))


class Renderer:

    def __init__(self, net):
        self.net = net

    def paint_neural_human(self, batch, holder_feat_map, holder_feat_scale,
                           prev_weight=None, prev_holder=None, use_pixel=False):

        smpl_vertice = torch.cat(batch['smpl_vertice'], dim=0)

        if cfg.rasterize:
            vizmap = torch.cat(batch['input_vizmaps'], dim=1)

        image_shape = batch['input_imgs'][0].shape[-2:]

        input_R = batch['input_R']
        input_T = batch['input_T']
        input_K = batch['input_K']

        input_R = input_R.reshape(-1, 3, 3)
        input_T = input_T.reshape(-1, 3, 1)
        input_K = input_K.reshape(-1, 3, 3)

        if cfg.rasterize:
            result = vizmap[0]
        else:
            result = None

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
                                      cfg.embed_size)).to(device=holder_feat_map.device) \
                if not use_pixel else torch.zeros((latent.shape[0], latent.shape[1], 3)).to(device=holder_feat_map.device)
            big_holder[final_result==True, :] = latent[final_result==True, :]

            if cfg.weight == 'cross_transformer':
                return final_result, big_holder

        else:

            holder = torch.sum(latent, dim=0, keepdim=True)
            holder = holder / num_input
            return result, holder

    def sample_from_feature_map(self, feat_map, feat_scale, image_shape, uv):

        scale = feat_scale / image_shape
        scale = torch.tensor(scale).to(dtype=torch.float32).to(
            device=feat_map.device)

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
        dhw = dhw - min_dhw[:, None, None]
        dhw = dhw / torch.tensor(cfg.voxel_size).to(dhw)
        # convert the voxel coordinate to [-1, 1]
        out_sh = torch.tensor(sp_input['out_sh']).to(dhw)
        dhw = dhw / out_sh * 2 - 1
        # convert dhw to whd, since the occupancy is indexed by dhw
        grid_coords = dhw[..., [2, 1, 0]]
        return grid_coords

    def batchify_rays(self,
                      sp_input,
                      grid_coords,
                      viewdir=None,
                      light_pts=None,
                      chunk=1024 * 32,
                      net_c=None,
                      batch=None,
                      xyz=None,
                      pixel_feat_map=None,
                      pixel_feat_scale=None,
                      norm_viewdir=None,
                      holder=None,
                      embed_xyz=None,
                      image_rgb=None,
                      smpl_dist=None,
                      cam_id=None):
        """Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = []

        for i in range(0, grid_coords.shape[1], chunk):

            xyz_shape = xyz.shape
            xyz = xyz.reshape(xyz_shape[0], -1, 3)
            pixel_feat = self.get_pixel_aligned_feature(batch,
                                                        xyz[:, i:i + chunk],
                                                        pixel_feat_map,
                                                        pixel_feat_scale)
            ret = self.net(pixel_feat, sp_input,
                           grid_coords[:, i:i + chunk],
                           holder=holder)#, cam_id=cam_id)#, dist=smpl_dist)
                           # image_rgb=image_rgb[:, :, i:i+chunk])

            all_ret.append(ret)
            if cfg.run_mode == 'test':
                gc.collect()
                torch.cuda.empty_cache()
        all_ret = torch.cat(all_ret, 1)

        return all_ret

    def transform_pts2frames(self, pbw, pts, batch, debug=False, batchify=False, chunk=1024 * 32):
        # function to transform points in the target space to the observed space
        A_target = batch['A_target']
        Rh = batch['R']
        Th = batch['Th']
        padding = torch.zeros([1, 1, 4]).to(device=A_target.device)
        padding[:, :, 3] = 1
        transforms_target = torch.cat([Rh, Th.transpose(1, 2)], dim=2)
        transforms_target = torch.cat([transforms_target, padding], dim=1)
        transforms_target_inv = torch.inverse(transforms_target.cpu())
        transforms_target_inv = transforms_target_inv.to(device=transforms_target.device)
        lbs_weights = pbw.transpose(1, 2)
        A_target = torch.matmul(lbs_weights, A_target.view(A_target.shape[0], 24, -1))
        A_target_inv = torch.inverse(A_target.view(A_target.shape[0], A_target.shape[1], 4, 4))

        Rh_in = batch['Rh_in'][0]
        Rh_in = batch_rodrigues(Rh_in.squeeze(1))
        Th_in = batch['Th_in'][0]
        A_in = batch['A_in'][0]
        transforms_t2p = torch.cat([Rh_in, Th_in.transpose(1, 2)], dim=2)
        padding = torch.zeros([transforms_t2p.shape[0], 1, 4]).to(device=transforms_t2p.device)
        padding[:, :, 3] = 1
        transforms_t2p = torch.cat([transforms_t2p, padding], dim=1)
        A_t2p = torch.matmul(lbs_weights, A_in.view(A_in.shape[0], 24, -1))
        A_t2p = A_t2p.view(A_t2p.shape[0], A_t2p.shape[1], 4, 4)
        if batchify:
            pts2frame = []
            for i in range(0, A_t2p.shape[1], chunk):
                A_t2p_b = A_t2p[:, i:i+chunk]
                A_target_inv_b = A_target_inv[:, i:i+chunk]
                transforms_c2n_b = transforms_t2p.unsqueeze(1) @ A_t2p_b @ A_target_inv_b @ transforms_target_inv.unsqueeze(1)
                pts_b = pts[:, i:i+chunk]
                padding = torch.ones(pts_b.shape[0], pts_b.shape[1], 1).to(device=pts.device)
                pts_homo_b = torch.cat([pts_b, padding], dim=-1)
                pts2frame_b = torch.matmul(transforms_c2n_b, pts_homo_b.unsqueeze(-1))
                pts2frame.append(pts2frame_b)
            pts2frame = torch.cat(pts2frame, dim=1)
        else:
            transforms_c2n = transforms_t2p.unsqueeze(1) @ A_t2p @ A_target_inv @ transforms_target_inv.unsqueeze(1)
            padding = torch.ones(pts.shape[0], pts.shape[1], 1).to(device=pts.device)
            pts_homo = torch.cat([pts, padding], dim=-1)
            pts2frame = torch.matmul(transforms_c2n, pts_homo.unsqueeze(-1))
        pts2frame = pts2frame[:, :, :3, 0]
        return pts2frame

    def render(self, batch):
        pts = batch['pts']
        sh = pts.shape
        light_pts = pts.clone()
        xyz = pts.clone()
        bw = batch['pbw']
        poses = batch['pose_target'][:, :, None]

        inside = batch['inside'][0].bool()

        pts = pts[0][inside][None]
        xyz = xyz[0][inside][None]
        # filter points that are far away from SMPL surface
        pts = pts.view(sh[0], -1, 1, 3)
        pts = self.pts_to_can_pts(pts, batch)
        pbw = bounds_grid_sample_blend_weights(pts.view(pts.shape[0], -1, 3), bw, batch['bounds'])
        n_batch = pbw.shape[0]
        n_point = pbw.shape[-1]
        pnorm = pbw[:, -1]
        pind = pnorm < cfg.norm_th
        pind[torch.arange(len(pnorm)), pnorm.argmin(dim=1)] = True
        pbw = pbw[:, :-1, pind[0]]
        pnorm = pnorm[:, pind[0]]
        pts = pts[pind][None]
        # refine SMPL based skinning weights
        if cfg.bw_refine:
            dist_embed = embedder.dist_embedder(pnorm[:, None].transpose(1, 2))
            if cfg.pose_encode:
                pbw = self.net.bw_refine(pbw, dist_embed.transpose(1, 2), poses.expand(-1, -1, pbw.shape[-1]))
            else:
                pbw = self.net.bw_refine(pbw, dist_embed.transpose(1, 2))

        xyz = xyz[pind][None]

        sp_input = self.prepare_sp_input(batch)
        grid_coords = self.get_grid_coords(pts, sp_input,
                                           batch)
        grid_coords = grid_coords.view(sh[0], -1, 3)
        # compute points in the observed space
        if grid_coords.size(1) <= 1024*32:
            xyz_transform = self.transform_pts2frames(pbw, xyz.view(xyz.shape[0], -1, 3), batch)
        else:
            xyz_transform = self.transform_pts2frames(pbw, xyz.view(xyz.shape[0], -1, 3), batch, batchify=True)
        ### --- get feature maps from encoder
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
            holder = torch.sum(holder * weight[..., None], dim=0, keepdim=True) / (torch.sum(weight[..., None], dim=0, keepdim=True) + 1e-8)
            holder_img = torch.sum(holder_img * weight[..., None], dim=0) / (
                    torch.sum(weight[..., None], dim=0) + 1e-8)
        else:
            holder = torch.mean(holder, dim=0, keepdim=True)
            holder_img = torch.mean(holder_img, dim=0, keepdim=True)

        if grid_coords.size(1) <= 1024*32:

            pixel_feat = self.get_pixel_aligned_feature(batch, xyz_transform,
                                                        pixel_feat_map,
                                                        pixel_feat_scale)

            alpha = self.net(pixel_feat, sp_input, grid_coords, holder=holder)#, cam_id=batch['cam_embid'])#, dist=smpl_dist)
                           # image_rgb=pixel_feat_image)

        else:
            alpha = self.batchify_rays(sp_input, grid_coords,
                                     chunk=1024 * 32, net_c=None,
                                     batch=batch, xyz=xyz_transform,
                                     pixel_feat_map=pixel_feat_map,
                                     pixel_feat_scale=pixel_feat_scale,
                                     holder=holder)#, cam_id=batch['cam_embid'])# smpl_dist=smpl_dist)
                                     # image_rgb=pixel_feat_image)

        raw_full = np.zeros([n_batch, n_point, 1])
        raw_full[pind.detach().cpu().numpy()] = alpha.detach().cpu().numpy()

        cube = np.zeros(sh[1:-1])
        inside = inside.detach().cpu().numpy()
        cube[inside == 1] = raw_full[0, :, 0]

        cube = np.pad(cube, 10, mode='constant')
        vertices, triangles = mcubes.marching_cubes(cube, cfg.mesh_th)
        mesh = trimesh.Trimesh(vertices, triangles)

        ret = {'cube': cube, 'mesh': mesh}

        return ret
