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
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as rotation
import random
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

        self.Ks = {}
        self.Rs = {}
        self.Ts = {}
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
            num_cams = len(self.cams[human]['K'])

            if human in ['CoreView_313', 'CoreView_315']:
                ims = np.array([
                    np.array([data_root + '/' + x.split('/')[0] + '/' +
                              x.split('/')[1].split('_')[4] + '.jpg' for x
                              in ims_data['ims']])[cfg.test_input_view]
                    for ims_data in annots['ims'][i:i + ni]
                ])
            else:
                ims = np.array([
                    np.array(
                        [data_root + '/' + x for x in ims_data['ims']])[
                        cfg.test_input_view]
                    for ims_data in annots['ims'][i:i + ni]
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


            self.Ks[human] = np.array(self.cams[human]['K'])[
                cfg.test_input_view].astype(
                np.float32)
            self.Rs[human] = np.array(self.cams[human]['R'])[
                cfg.test_input_view].astype(
                np.float32)
            self.Ts[human] = np.array(self.cams[human]['T'])[
                                 cfg.test_input_view].astype(
                np.float32) / 1000.
            self.Ds[human] = np.array(self.cams[human]['D'])[
                cfg.test_input_view].astype(
                np.float32)

            # self.ni = cfg.ni

        self.ims = np.array(self.ims)
        self.nrays = cfg.N_rand

    def image_to_tensor(self):

        ops = []
        ops.extend(
            [transforms.ToTensor(), ]
        )
        return transforms.Compose(ops)

    def get_input_mask(self, human, index, filename):
        # index is camera index

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

        # msk_path = os.path.join(self.data_root, 'mask_cihp',
        #                        'Camera_B'+str(index),filename[:-4]+'.png')
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

    def get_mask(self, index):

        data_info = self.ims[index].split('/')
        human = data_info[-3]
        camera = data_info[-2]
        frame = data_info[-1]

        msk_exist = False
        msk_cihp_exist = False

        msk_path = os.path.join(cfg.virt_data_root, human, 'mask',
                                camera, frame)[:-4] + '.png'
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

        # border = 3
        # kernel = np.ones((border, border), np.uint8)
        # msk = cv2.dilate(msk.copy(), kernel)
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

        # obtain the original bounds for point sampling
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


        Rh = params['Rh']  # (1,3)
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)  # (3,3)
        Th = params['Th'].astype(np.float32)  # (1,3)
        poses = params['poses'].reshape(-1, 3).astype(np.float32)

        xyz = np.dot(xyz - Th, R)

        # transformation augmentation
        xyz, center, rot, trans = if_nerf_dutils.transform_can_smpl(xyz)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)


        if cfg.big_box:
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

        return feature, coord, out_sh, can_bounds, bounds, R, Th, center, rot, trans, smpl_vertices, poses

    def get_mask(self, i, nv):

        im = self.ims[i, nv]

        data_info = im.split('/')
        human = data_info[-3]
        camera = data_info[-2]
        frame = data_info[-1]

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

        # msk = (msk | msk_cihp).astype(np.uint8)

        if msk_exist and msk_cihp_exist:
            msk = (msk | msk_cihp).astype(np.uint8)
        elif msk_exist and not msk_cihp_exist:
            msk = msk.astype(np.uint8)
        elif not msk_exist and msk_cihp_exist:
            msk = msk_cihp.astype(np.uint8)

        msk = cv2.undistort(msk, self.Ks[human][nv], self.Ds[human][nv])

        border = 5
        kernel = np.ones((border, border), np.uint8)
        msk = cv2.dilate(msk.copy(), kernel)

        return msk

    def prepare_inside_pts(self, pts, i):


        human = (self.ims[i][0]).split('/')[-3]

        sh = pts.shape
        pts3d = pts.reshape(-1, 3)

        inside = np.ones([len(pts3d)]).astype(np.uint8)

        for nv in range(self.ims.shape[1]):
            ind = inside == 1
            pts3d_ = pts3d[ind]

            RT = np.concatenate([self.Rs[human][nv], self.Ts[human][nv]],
                                axis=1)
            pts2d = base_utils.project(pts3d_, self.Ks[human][nv], RT)

            msk = self.get_mask(i, nv)
            H, W = msk.shape
            pts2d = np.round(pts2d).astype(np.int32)
            pts2d[:, 0] = np.clip(pts2d[:, 0], 0, W - 1)
            pts2d[:, 1] = np.clip(pts2d[:, 1], 0, H - 1)
            msk_ = msk[pts2d[:, 1], pts2d[:, 0]]

            inside[ind] = msk_

        inside = inside.reshape(*sh[:-1])

        return inside

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

        cam_ind = self.cam_inds[index]
        smpl_target = self.get_smpl_vertice(human, int(frame[:-4]))
        R = np.array(self.cams[human]['R'][cam_ind])
        # when generating the annots.py, 1000 is multiplied, so dividing back
        T = np.array(self.cams[human]['T'][cam_ind]) / 1000.
        smpl_target_c = np.matmul(R, smpl_target.transpose()) + T
        pbw = np.load(os.path.join(cfg.virt_data_root, human, cfg.lbs, 'bweights/{}.npy'.format(int(frame[:-4]))))
        pbw = pbw.astype(np.float32)

        num_inputs = len(cfg['test_input_view'])

        lbs_root = os.path.join(cfg.virt_data_root, human, cfg.lbs)
        joints_t = np.load(os.path.join(lbs_root, 'joints.npy'))
        joints_t = joints_t.astype(np.float32)
        parents = np.load(os.path.join(lbs_root, 'parents.npy'))

        feature, coord, out_sh, can_bounds, bounds, Rh, Th, center, rot, trans, tmp_smpl_vertices, poses = self.prepare_input(
            human, i)
        poses_target = poses.copy().reshape(-1)[3:] + 1e-3
        poses_target = torch.from_numpy(poses_target.astype(np.float32))

        voxel_size = cfg.voxel_size
        x = np.arange(can_bounds[0, 0], can_bounds[1, 0] + voxel_size[0],
                      voxel_size[0])
        y = np.arange(can_bounds[0, 1], can_bounds[1, 1] + voxel_size[1],
                      voxel_size[1])
        z = np.arange(can_bounds[0, 2], can_bounds[1, 2] + voxel_size[2],
                      voxel_size[2])
        pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
        pts = pts.astype(np.float32)

        inside = self.prepare_inside_pts(pts, index)

        ### --- prepare input images for image feature extraction
        input_vizmaps = []
        input_imgs = []
        input_msks = []
        input_K = []
        input_R = []
        input_T = []
        smpl_vertices = []
        if cfg.time_steps == 1:
            smpl_vertices.append(tmp_smpl_vertices)

        filename = frame

        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)

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

        A_target = if_nerf_dutils.get_rigid_transformation(poses, joints_t, parents)

        A_target = torch.from_numpy(A_target)
        Rh = torch.from_numpy(Rh)
        Th = torch.from_numpy(Th)

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

                input_img = imageio.imread(input_img_path).astype(
                    np.float32) / 255.

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
                                          kernel)  # [0,255]

                in_K[:2] = in_K[:2] * cfg.ratio

                input_img = self.im2tensor(input_img)
                input_msk = self.im2tensor(input_msk).bool()

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

        i = int(frame[:-4])

        ret = {
            'smpl_vertice': smpl_vertices,
            'feature': feature,
            'coord': coord,
            'out_sh': out_sh,
            'pts': pts,
            'inside': inside,
            'vertices_target': smpl_target,
            'pbw': pbw,
            'lbs_weights': self.lbs_weights,
        }
        human_idx = 0
        if self.split == 'test':
            human_idx = self.human_idx_name[human]

        meta = {
            'human_idx': human_idx,
            'frame_index': frame_index,
            'bounds': bounds,
            'R': Rh,
            'Th': Th,
            'center': center,
            'rot': rot,
            'trans': trans,
            'i': i,
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
            'pose_target': poses_target
            # 'target_image': img
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.ims)