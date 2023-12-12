import torch
import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
from torchvision import transforms
import numpy as np
import json
import os
import imageio
import cv2
from lib.config import cfg
from lib.datasets import get_human_info
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from plyfile import PlyData
from lib.utils import render_utils
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as rotation
import random
import pdb
import pickle

class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()
        with open('./data/smplx/smpl/SMPL_NEUTRAL.pkl', 'rb') as f:
            smpl_info = pickle.load(f, encoding="latin1")
        self.lbs_weights = smpl_info['weights'].astype(np.float32)
        self.split = split
        self.im2tensor = self.image_to_tensor()
        self.cams = {}
        self.ims = []
        self.cam_inds = []

        self.K = {}
        self.render_w2c = {}
        self.Ks = {}
        self.RT = {}
        self.Ds = {}

        self.start_end = {}

        data_name = cfg.virt_data_root.split('/')[-1]
        human_info = get_human_info.get_human_info(self.split)
        human_list = list(human_info.keys())

        if self.split == 'test':
            self.human_idx_name = {}
            for human_idx in range(len(human_list)):
                human = human_list[human_idx]
                self.human_idx_name[human] = human_idx

        self.select_frames = {}
        for idx in range(len(human_list)):

            human = human_list[idx]

            data_root = os.path.join(cfg.virt_data_root, human)

            ann_file = os.path.join(cfg.virt_data_root, human, 'annots.npy')
            annots = np.load(ann_file, allow_pickle=True).item()
            self.cams[human] = annots['cams']

            i = 0
            i = i + human_info[human]['begin_i']
            ni = human_info[human]['ni']

            end = i + ni
            num_frame = cfg.time_steps
            interval = int((ni-1) / num_frame)
            select_frame = np.arange(i+1, end, interval, dtype=int)[:num_frame]
            self.select_frames[human] = select_frame

            K, RT = render_utils.load_cam(ann_file)

            render_w2c = render_utils.gen_path_virt(RT, render_views=(ni))

            if human in ['CoreView_313', 'CoreView_315']:

                ims = np.array([
                    np.array([data_root + '/' + x.split('/')[0] + '/' +
                              x.split('/')[1].split('_')[4] + '.jpg' for x
                              in ims_data['ims']])[cfg.test_input_view]
                    for ims_data in annots['ims'][i: i + ni]
                ])
            else:
                ims = np.array([
                    np.array(
                        [data_root + '/' + x for x in ims_data['ims']])[
                        cfg.test_input_view]
                    for ims_data in annots['ims'][i: i + ni]
                ])

            cam_inds = np.array([
                np.arange(len(ims_data['ims']))[cfg.test_input_view]
                for ims_data in annots['ims'][i:i + ni]
            ]).ravel()

            start_idx = len(self.ims)
            length = len(ims)
            self.ims.extend(ims)
            self.cam_inds.extend(cam_inds)

            self.start_end[human] = {}
            self.start_end[human]['start'] = int(
                self.ims[start_idx][0].split('/')[-1][:-4])
            self.start_end[human]['end'] = int(
                self.ims[start_idx + length - 1][0].split('/')[-1][:-4])
            self.start_end[human]['length'] = self.start_end[human]['end'] - \
                                              self.start_end[human]['start']

            self.start_end[human]['intv'] = human_info[human]['i_intv']

            self.K[human] = K[0]
            self.render_w2c[human] = render_w2c

            self.Ks[human] = np.array(K)[cfg.test_input_view].astype(np.float32)
            self.RT[human] = np.array(RT)[cfg.test_input_view].astype(
                np.float32)
            self.Ds[human] = np.array(self.cams[human]['D'])[
                cfg.test_input_view].astype(np.float32)


    def image_to_tensor(self):

        ops = []
        ops.extend(
            [transforms.ToTensor(), ]
        )
        return transforms.Compose(ops)

    def get_input_mask(self, human, index, filename):

        msk_exist = False
        msk_cihp_exist = False

        if human in ['CoreView_313', 'CoreView_315']:
            msk_path = os.path.join(cfg.virt_data_root, human, 'mask',
                                    'Camera (' + str(index) + ')',
                                    filename[:-4] + '.png')
        else:
            msk_path = os.path.join(cfg.virt_data_root, human, 'mask',
                                    'Camera_B' + str(index),
                                    filename[:-4] + '.png')

        msk_exist = os.path.exists(msk_path)

        if msk_exist:
            msk = imageio.imread(msk_path)
            msk = (msk != 0).astype(np.uint8)

        if human in ['CoreView_313', 'CoreView_315']:
            msk_path = os.path.join(cfg.virt_data_root, human, 'mask_cihp',
                                    'Camera (' + str(index) + ')',
                                    filename[:-4] + '.png')
        else:
            msk_path = os.path.join(cfg.virt_data_root, human, 'mask_cihp',
                                    'Camera_B' + str(index),
                                    filename[:-4] + '.png')
        msk_cihp_exist = os.path.exists(msk_path)

        if msk_cihp_exist:
            msk_cihp = imageio.imread(msk_path)
            msk_cihp = (msk_cihp != 0).astype(np.uint8)

        if msk_exist and msk_cihp_exist:
            msk = (msk | msk_cihp).astype(np.uint8)

        elif msk_exist and not msk_cihp_exist:
            msk = msk.astype(np.uint8)

        elif not msk_exist and msk_cihp_exist:
            msk = msk_cihp.astype(np.uint8)

        return msk

    def get_smpl_vertice(self, human, frame):

        vertices_path = os.path.join(cfg.virt_data_root, human, 'vertices',
                                     '{}.npy'.format(frame))
        smpl_vertice = np.load(vertices_path).astype(np.float32)

        return smpl_vertice

    def prepare_input(self, human, i):

        vertices_path = os.path.join(cfg.virt_data_root, human, 'vertices',
                                     '{}.npy'.format(i))

        xyz = np.load(vertices_path).astype(np.float32)
        smpl_vertices = None
        if cfg.time_steps == 1:
            smpl_vertices = np.array(xyz)

        nxyz = np.zeros_like(xyz).astype(np.float32)

        # obtain the origin bounds for point sampling
        data_name = cfg.virt_data_root.split('/')[-1]
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)

        if cfg.big_box:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05


        can_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(cfg.virt_data_root, human, cfg.params,
                                   '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        data_name = cfg.virt_data_root.split('/')[-1]

        Rh = params['Rh']
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = params['Th'].astype(np.float32)
        poses = params['poses'].reshape(-1, 3).astype(np.float32)

        xyz = np.dot(xyz - Th, R)

        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)

        if cfg.big_box:  # False
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05

        bounds = np.stack([min_xyz, max_xyz], axis=0)

        cxyz = xyz.astype(np.float32)
        nxyz = nxyz.astype(np.float32)
        feature = np.concatenate([cxyz, nxyz], axis=1).astype(np.float32)

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array(cfg.voxel_size)
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        return feature, coord, out_sh, can_bounds, bounds, R, Th, smpl_vertices, poses

    def get_mask(self, i):

        ims = self.ims[i]
        msks = []

        for nv in range(len(ims)):
            im = ims[nv]

            data_info = im.split('/')
            human = data_info[-3]
            camera = data_info[-2]
            frame = data_info[-1]

            msk_exist = False
            msk_cihp_exist = False

            msk_path = os.path.join(cfg.virt_data_root, human, 'mask', camera,
                                    frame)[:-4] + '.png'
            msk_exist = os.path.exists(msk_path)
            if msk_exist:
                msk = imageio.imread(msk_path)
                msk = (msk != 0).astype(np.uint8)

            msk_path = os.path.join(cfg.virt_data_root, human, 'mask_cihp',
                                    camera, frame)[:-4] + '.png'

            msk_cihp_exist = os.path.exists(msk_path)
            if msk_cihp_exist:
                msk_cihp = imageio.imread(msk_path)
                msk_cihp = (msk_cihp != 0).astype(np.uint8)


            if msk_exist and msk_cihp_exist:
                msk = (msk | msk_cihp).astype(np.uint8)
            elif msk_exist and not msk_cihp_exist:
                msk = msk.astype(np.uint8)
            elif not msk_exist and msk_cihp_exist:
                msk = msk_cihp.astype(np.uint8)

            K = self.Ks[human][nv].copy()

            K[:2] = K[:2] / cfg.ratio
            msk = cv2.undistort(msk, K, self.Ds[human][nv])

            data_name = cfg.virt_data_root.split('/')[-1]

            border = 5

            kernel = np.ones((border, border), np.uint8)
            msk = cv2.dilate(msk.copy(), kernel)

            msks.append(msk)

        return msks

    def get_item(self, index):
        return self.__getitem__(index)

    def __getitem__(self, index):

        data_name = cfg.virt_data_root.split('/')[-1]
        img_path = self.ims[index][0]

        data_info = img_path.split('/')
        human = data_info[-3]
        camera = data_info[-2]
        frame = data_info[-1]

        i = int(frame[:-4])

        cam_ind = i % len(self.render_w2c[human])
        smpl_target = self.get_smpl_vertice(human, int(frame[:-4]))
        R = self.render_w2c[human][cam_ind][:3, :3]
        # when generating the annots.py, 1000 is multiplied, so dividing back
        T = self.render_w2c[human][cam_ind][:3, 3:]
        smpl_target_c = np.matmul(R, smpl_target.transpose()) + T
        pbw = np.load(os.path.join(cfg.virt_data_root, human, cfg.lbs, 'bweights/{}.npy'.format(int(frame[:-4]))))
        pbw = pbw.astype(np.float32)

        # camera embedding cannot be used here because the use of virtual cameras
        if cfg.camera_embedding:
            if human in ['CoreView_313', 'CoreView_315']:
                cam_idx_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                14, 15, 16, 17, 18, 21, 22]
                cam_embid = cam_idx_list[cam_ind]
            else:
                cam_embid = self.cam_inds[index]
        else:
            cam_embid = 0

        lbs_root = os.path.join(cfg.virt_data_root, human, cfg.lbs)
        joints_t = np.load(os.path.join(lbs_root, 'joints.npy'))
        joints_t = joints_t.astype(np.float32)
        parents = np.load(os.path.join(lbs_root, 'parents.npy'))

        feature, coord, out_sh, can_bounds, bounds, Rh, Th, tmp_smpl_vertices, poses = self.prepare_input(
            human, i)

        poses_target = poses.copy().reshape(-1)[3:] + 1e-3
        poses_target = torch.from_numpy(poses_target.astype(np.float32))

        A_target = if_nerf_dutils.get_rigid_transformation(poses, joints_t, parents)

        A_target = torch.from_numpy(A_target)
        Rh = torch.from_numpy(Rh)
        Th = torch.from_numpy(Th)

        ### --- prepare input masks for constructing visual hull
        msks = self.get_mask(index)

        # reduce the image resolution by ratio
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        msks = [
            cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            for msk in msks
        ]
        msks = np.array(msks)

        ### --- sample rays from the virtual target camera view

        K = self.K[human]


        ray_o, ray_d, near, far, center, scale, mask_at_box = render_utils.image_rays(
            self.render_w2c[human][cam_ind], K, can_bounds)

        ### --- prepare input images for image feature extraction
        input_vizmaps = []
        input_imgs = []
        input_msks = []
        input_K = []
        input_R = []
        input_T = []
        smpl_vertices = []

        if human in ['CoreView_313', 'CoreView_315']:
            cam_idx_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                            15, 16, 17, 18, 21, 22]
        input_view = cfg.test_input_view

        target_frame = frame[:-4]
        frame_index = int(target_frame)
        zfill = len(target_frame)

        idx = input_view[0]
        in_R = np.array(self.cams[human]['R'][idx]).astype(np.float32)
        in_T = (np.array(self.cams[human]['T'][idx]) / 1000.).astype(
            np.float32)
        smpl_vertices_path = os.path.join(cfg.virt_data_root, human, 'vertices.npy')
        smpl_verts = np.load(smpl_vertices_path).astype(np.float32)
        smpl_vertices_c = np.matmul(in_R[None, ...], np.transpose(smpl_verts, (0, 2, 1))) + in_T[None, ...]
        smpl_dist = np.sum(np.linalg.norm(smpl_target_c[None, ...] - smpl_vertices_c, axis=1), axis=1)
        smpl_index = np.argsort(smpl_dist)
        smpl_index = smpl_index[:cfg.time_steps]

        if cfg.evenly_select:
            frame_ids = self.select_frames[human]
        else:
            if human in ['CoreView_313', 'CoreView_315']:
                frame_ids = smpl_index * 5 + 1
            else:
                frame_ids = smpl_index * 5

        A_in = []
        Rh_in = []
        Th_in = []
        for t in range(cfg.time_steps):
            current_frame = frame_ids[t]
            filename = str(current_frame).zfill(zfill) + '.jpg'

            if cfg.time_steps > 1:
                smpl_vertices.append(
                    self.get_smpl_vertice(human, current_frame))

            tmp_vizmaps = []
            tmp_imgs = []
            tmp_msks = []

            for j in range(len(cfg['test_input_view'])):
                idx = cfg['test_input_view'][j]
                cam_idx = None
                if human in ['CoreView_313', 'CoreView_315']:
                    cam_idx = cam_idx_list[idx]

                if human in ['CoreView_313', 'CoreView_315']:
                    input_img_path = os.path.join(cfg.virt_data_root, human,
                                                  'Camera (' + str(
                                                      cam_idx + 1) + ')',
                                                  filename)
                else:
                    input_img_path = os.path.join(cfg.virt_data_root, human,
                                                  'Camera_B' + str(idx + 1),
                                                  filename)

                input_img = imageio.imread(input_img_path).astype(
                    np.float32) / 255.

                if cfg.rasterize:
                    vizmap_idx = str(current_frame).zfill(zfill)
                    if human in ['CoreView_313', 'CoreView_315']:
                        vizmap_path = os.path.join(cfg.rasterize_root, human,
                                                   'visibility',
                                                   'Camera (' + str(
                                                       cam_idx + 1) + ')',
                                                   '{}.npy'.format(vizmap_idx))
                    else:
                        vizmap_path = os.path.join(cfg.rasterize_root, human,
                                                   'visibility',
                                                   'Camera_B' + str(idx + 1),
                                                   '{}.npy'.format(vizmap_idx))
                    input_vizmap = np.load(vizmap_path).astype(np.bool)

                if human in ['CoreView_313', 'CoreView_315']:
                    input_msk = self.get_input_mask(human, cam_idx + 1,
                                                    filename)
                else:
                    input_msk = self.get_input_mask(human, idx + 1, filename)
                in_K = np.array(self.cams[human]['K'][idx]).astype(np.float32)
                in_D = np.array(self.cams[human]['D'][idx]).astype(np.float32)

                input_img = cv2.undistort(input_img, in_K, in_D)
                input_msk = cv2.undistort(input_msk, in_K, in_D)

                in_R = np.array(self.cams[human]['R'][idx]).astype(np.float32)
                in_T = (np.array(self.cams[human]['T'][idx]) / 1000.).astype(
                    np.float32)

                input_img = cv2.resize(input_img, (W, H),
                                       interpolation=cv2.INTER_AREA)
                input_msk = cv2.resize(input_msk, (W, H),
                                       interpolation=cv2.INTER_NEAREST)
                if cfg.mask_bkgd:
                    if cfg.white_bkgd:
                        input_img[input_msk == 0] = 1
                    else:
                        input_img[input_msk == 0] = 0

                input_msk = (
                            input_msk != 0)
                if cfg.use_viz_test and cfg.use_fg_masking:
                    if cfg.ratio == 0.5:
                        border = 5

                    kernel = np.ones((border, border), np.uint8)
                    input_msk = cv2.erode(input_msk.astype(np.uint8) * 255,
                                          kernel)

                input_img = self.im2tensor(input_img)
                input_msk = self.im2tensor(input_msk).bool()

                in_K[:2] = in_K[:2] * cfg.ratio

                tmp_imgs.append(input_img)
                tmp_msks.append(input_msk)
                if cfg.rasterize:
                    tmp_vizmaps.append(torch.from_numpy(input_vizmap))
                if t == 0:
                    input_K.append(torch.from_numpy(in_K))
                    input_R.append(torch.from_numpy(in_R))
                    input_T.append(torch.from_numpy(in_T))

            input_imgs.append(torch.stack(tmp_imgs))
            input_msks.append(torch.stack(tmp_msks))
            if cfg.rasterize:
                input_vizmaps.append(torch.stack(tmp_vizmaps))

            params_path = os.path.join(cfg.virt_data_root, human, cfg.params,
                                       '{}.npy'.format(current_frame))
            params = np.load(params_path, allow_pickle=True).item()
            poses = params['poses'].reshape(-1, 3)
            Rh_input = params['Rh'].astype(np.float32)
            Th_input = params['Th'].astype(np.float32)

            A = if_nerf_dutils.get_rigid_transformation(poses, joints_t, parents)
            A_in.append(torch.from_numpy(A))
            Rh_in.append(torch.from_numpy(Rh_input))
            Th_in.append(torch.from_numpy(Th_input))

        input_K = torch.stack(input_K)
        input_R = torch.stack(input_R)
        input_T = torch.stack(input_T)

        A_in = torch.stack(A_in)
        Rh_in = torch.stack(Rh_in)
        Th_in = torch.stack(Th_in)

        ret = {
            'smpl_vertice': smpl_vertices,
            'feature': feature,
            'coord': coord,
            'out_sh': out_sh,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box,
            'vertices_target': smpl_target,
            'pbw': pbw,
            'lbs_weights': self.lbs_weights,
        }

        i = int(frame[:-4])
        human_idx = 0
        if self.split == 'test':
            human_idx = self.human_idx_name[human]
        meta = {
            'human_idx': human_idx,
            'frame_index': frame_index,
            'bounds': bounds,
            'R': Rh,
            'Th': Th,
            'i': i,
            'index': index,
            'input_imgs': input_imgs,
            'input_msks': input_msks,
            'input_vizmaps': input_vizmaps,
            'input_K': input_K,
            'input_R': input_R,
            'input_T': input_T,
            'Rh_in': Rh_in,
            'Th_in': Th_in,
            'A_target': A_target,
            'A_in': A_in,
            'cam_embid': cam_embid,
            'pose_target': poses_target
        }

        ret.update(meta)

        meta = {'msks': msks, 'Ks': self.Ks[human], 'RT': self.RT[human]}
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.ims)
