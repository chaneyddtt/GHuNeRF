import sys
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
from lib.datasets import get_human_info_snapshot
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
import pickle
from plyfile import PlyData
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as rotation
import pdb
import random
import time
from zju_smpl.smplmodel.lbs import lbs, batch_rodrigues
from lib.utils.snapshot_data_utils import get_camera


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        with open('./data/smplx/smpl/SMPL_NEUTRAL.pkl', 'rb') as f:
            smpl_info = pickle.load(f, encoding="latin1")
        self.lbs_weights = smpl_info['weights'].astype(np.float32)
        self.split = split  # 'train'
        self.im2tensor = self.image_to_tensor()

        self.cams = {}
        self.ims = []
        self.cam_inds = []
        self.start_end = {}
        self.select_frames = {}

        self.Ks = {}
        self.Rs = {}
        self.Ts = {}
        self.Ds = {}

        data_name = cfg.virt_data_root.split('/')[-1]
        human_info = get_human_info_snapshot.get_human_info(self.split)
        human_list = list(human_info.keys())

        if self.split == 'test':
            self.human_idx_name = {}
            for human_idx in range(len(human_list)):
                human = human_list[human_idx]
                self.human_idx_name[human] = human_idx

        for idx in range(len(human_list)):
            human = human_list[idx]

            data_root = os.path.join(cfg.virt_data_root, human)
            camera_path = os.path.join(data_root, 'camera.pkl')
            self.cams[human] = get_camera(camera_path)

            i = 0
            i = i + human_info[human]['begin_i']
            i_inv = human_info[human]['i_intv']
            ni = human_info[human]['ni']

            end = i + ni
            num_frame = cfg.time_steps
            interval = int((ni - 1) / num_frame)
            select_frame = np.arange(i, end, interval, dtype=int)[:num_frame]

            image_folder = os.path.join(data_root, 'image')
            # ims = sorted(os.listdir(image_folder))
            ims = os.listdir(image_folder)
            ims = sorted([im.split('.')[0].zfill(3) + '.jpg' for im in ims])
            if cfg.run_mode == 'test':
                all_frames = range(i, i+ni, i_inv)
                all_frames = [*all_frames]
                test_frames = list(set(all_frames) - set(select_frame))
                ims_ = [ims[f] for f in test_frames]
            else:
                ims_ = ims[i:i+ni][::i_inv]

            start_idx = len(self.ims)
            length = len(ims_)
            self.ims.extend(ims_)
            self.ims[start_idx:start_idx+length] = [
                    image_folder + '/' + x for x in
                    self.ims[start_idx:start_idx + length]]

            self.start_end[human] = {}
            self.start_end[human]['start'] = i
            self.start_end[human]['end'] = i + len(ims_)
            self.start_end[human]['length'] = self.start_end[human]['end'] - self.start_end[human]['start']
            self.select_frames[human] = select_frame

            self.Ks[human] = np.array(self.cams[human]['K']).astype(np.float32)
            self.Rs[human] = np.array(self.cams[human]['R']).astype(np.float32)
            self.Ts[human] = np.array(self.cams[human]['T']).astype(np.float32)
            self.Ds[human] = np.array(self.cams[human]['D']).astype(np.float32)

        self.nrays = cfg.N_rand
        self.ims = np.array(self.ims)
        self.num_humans = len(human_list)

    def image_to_tensor(self):

        ops = []

        ops.extend(
            [transforms.ToTensor(), ]
        )

        return transforms.Compose(ops)

    def color_jitter(self):
        ops = []

        ops.extend(
            [transforms.ColorJitter(brightness=(0.5, 1.5),
                                    contrast=(0.5, 1.5), saturation=(0.5, 1.5),
                                    hue=(-0.2, 0.2)), ]
        )

        return transforms.Compose(ops)

    def get_mask(self, index):

        data_info = self.ims[index].split('/')
        human = data_info[-3]
        frame = data_info[-1]

        msk_path = os.path.join(cfg.virt_data_root, human, 'mask',
                                '{}.png'.format(int(frame[:-4])))
        msk = imageio.imread(msk_path)
        msk = (msk != 0).astype(np.uint8)

        return msk

    def get_input_mask(self, human, filename):
        # index is camera index

        msk_path = os.path.join(cfg.virt_data_root, human, 'mask',
                                filename[:-4] + '.png')
        msk = imageio.imread(msk_path)
        msk = (msk != 0).astype(np.uint8)

        return msk

    def get_smpl_vertice(self, human, frame):

        vertices_path = os.path.join(cfg.virt_data_root, human, 'vertices',
                                     '{}.npy'.format(frame))
        smpl_vertice = np.load(vertices_path).astype(np.float32)

        return smpl_vertice

    def get_depth_map(self, human, frame):

        depth_map_path = os.path.join('./data/people_snapshot_depth', human, '{}.npy'.format(frame))
        depth_map = np.load(depth_map_path).astype(np.float32)

        return depth_map

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
        params_path = os.path.join(cfg.virt_data_root, human, 'params.npy')

        params = np.load(params_path, allow_pickle=True).item()


        Rh = params['pose'][i][:3]
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = params['trans'][i][None].astype(np.float32)
        poses = params['pose'][i].reshape(-1, 3).astype(np.float32)
        poses[0, :] = 0.

        xyz = np.dot(xyz - Th, R)
        # transformation augmentation
        xyz, center, rot, trans = if_nerf_dutils.transform_can_smpl(xyz)

        # obtain the bounds for coord construction
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

        return feature, coord, out_sh, can_bounds, bounds, R, Th, center, rot, trans, smpl_vertices, poses

    def prepare_inside_pts(self, pts, i):


        human = (self.ims[i]).split('/')[-3]

        sh = pts.shape
        pts3d = pts.reshape(-1, 3)

        inside = np.ones([len(pts3d)]).astype(np.uint8)

        for nv in range(1):
            ind = inside == 1
            pts3d_ = pts3d[ind]

            RT = np.concatenate([self.Rs[human], self.Ts[human][:, None]],
                                axis=1)
            pts2d = base_utils.project(pts3d_, self.Ks[human], RT)

            msk = self.get_mask(i)
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

        img_path = self.ims[index]

        data_info = img_path.split('/')
        human = data_info[-3]
        frame = data_info[-1]
        img_path = os.path.join(cfg.virt_data_root, human, 'image', '{}.jpg'.format(int(frame[:-4])))

        # img = imageio.imread(img_path)
        # img = img.astype(np.float32) / 255.
        smpl_target = self.get_smpl_vertice(human, int(frame[:-4]))
        pbw = np.load(os.path.join(cfg.virt_data_root, human, cfg.lbs, 'bweights/{}.npy'.format(int(frame[:-4]))))
        pbw = pbw.astype(np.float32)


        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)

        lbs_root = os.path.join(cfg.virt_data_root, human, cfg.lbs)
        joints_t = np.load(os.path.join(lbs_root, 'joints.npy'))
        joints_t = joints_t.astype(np.float32)
        parents = np.load(os.path.join(lbs_root, 'parents.npy'))

        target_frame = frame[:-4]
        frame_index = int(target_frame)

        start = self.start_end[human]['start']
        end = self.start_end[human]['end']
        length = self.start_end[human]['length']

        num_frame = cfg.time_steps
        interval = int(length/num_frame)
        select_frame = self.select_frames[human]

        i = int(frame[:-4])

        feature, coord, out_sh, can_bounds, bounds, Rh, Th, center, rot, trans, tmp_smpl_vertices, poses = \
                                                        self.prepare_input(human, i)

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

        input_vizmaps = []
        input_imgs = []
        input_msks = []
        input_K = []
        input_R = []
        input_T = []
        smpl_vertices = []


        if cfg.time_steps == 1:
            smpl_vertices.append(tmp_smpl_vertices)

        A_target = if_nerf_dutils.get_rigid_transformation(poses, joints_t, parents)

        A_target = torch.from_numpy(A_target)
        Rh = torch.from_numpy(Rh)
        Th = torch.from_numpy(Th)

        A_in = []
        Rh_in = []
        Th_in = []
        for t in range(num_frame):
            current_frame = select_frame[t]
            filename = str(current_frame) + '.jpg'

            if cfg.time_steps > 1:
                smpl_vertices.append(
                    self.get_smpl_vertice(human, current_frame))

            tmp_vizmaps = []
            tmp_imgs = []
            tmp_msks = []

            input_img_path = os.path.join(cfg.virt_data_root, human, 'image', filename)
            input_img = imageio.imread(input_img_path)
            input_img = input_img.astype(np.float32) / 255.

            if cfg.rasterize:
                vizmap_path = os.path.join(cfg.rasterize_root, human, str(current_frame) + '.npy')
                input_vizmap = np.load(vizmap_path).astype(np.bool)

            input_msk = self.get_input_mask(human, filename)

            in_K = np.array(self.cams[human]['K']).astype(np.float32)
            in_D = np.array(self.cams[human]['D']).astype(np.float32)

            input_img = cv2.undistort(input_img, in_K, in_D)
            input_msk = cv2.undistort(input_msk, in_K, in_D)

            in_R = np.array(self.cams[human]['R']).astype(np.float32)
            in_T =np.array(self.cams[human]['T'][:, None]).astype(np.float32)

            input_img = cv2.resize(input_img, (W, H), interpolation=cv2.INTER_AREA)
            input_msk = cv2.resize(input_msk, (W, H), interpolation=cv2.INTER_NEAREST)

            if cfg.mask_bkgd:
                if cfg.white_bkgd:
                    input_img[input_msk == 0] = 1
                else:
                    input_img[input_msk == 0] = 0

            input_msk = (input_msk != 0)  # bool mask : foreground (True) background (False)

            if cfg.use_viz_test and cfg.use_fg_masking:
                if cfg.ratio == 0.5:
                    border = 5
                kernel = np.ones((border, border), np.uint8)

                input_msk = cv2.erode(input_msk.astype(np.uint8) * 255,
                                      kernel)

            # [-1,1]
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

            params_path = os.path.join(cfg.virt_data_root, human, 'params.npy')
            params = np.load(params_path, allow_pickle=True).item()
            poses = params['pose'][current_frame].reshape(-1, 3)
            Rh_input = poses[0].copy().astype(np.float32)
            Th_input = params['trans'][current_frame].astype(np.float32)
            poses[0, :] = 0.

            A = if_nerf_dutils.get_rigid_transformation(poses, joints_t, parents)
            A_in.append(torch.from_numpy(A))
            Rh_in.append(torch.from_numpy(Rh_input[None]))
            Th_in.append(torch.from_numpy(Th_input[None]))
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
            'pts': pts,
            'inside': inside,
            'vertices_target': smpl_target,
            'pbw': pbw,
            'lbs_weights': self.lbs_weights,
        }

        i = int(os.path.basename(img_path)[:-4])
        human_idx = 0
        if self.split == 'test':
            human_idx = self.human_idx_name[human]
        meta = {
            'bounds': bounds,
            'R': Rh,
            'Th': Th,
            'center': center,
            'rot': rot,
            'trans': trans,
            'i': i,
            'human_idx': human_idx,
            'frame_index': frame_index,
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

        }
        ret.update(meta)

        return ret

    def get_length(self):
        return self.__len__()

    def __len__(self):
        return len(self.ims)
