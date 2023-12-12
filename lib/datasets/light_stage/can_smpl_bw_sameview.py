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
import pickle
from plyfile import PlyData
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as rotation
import pdb
import random
import time
from zju_smpl.smplmodel.lbs import lbs, batch_rodrigues


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()
        print("Dataset wit the same view")
        with open('./data/smplx/smpl/SMPL_NEUTRAL.pkl', 'rb') as f:
            smpl_info = pickle.load(f, encoding="latin1")
        self.lbs_weights = smpl_info['weights'].astype(np.float32)
        self.split = split  # 'train'
        self.im2tensor = self.image_to_tensor()
        if self.split == 'train' and cfg.jitter:
            self.jitter = self.color_jitter()

        self.cams = {}
        self.ims = []
        self.cam_inds = []
        self.start_end = {}
        self.select_frames = {}

        data_name = cfg.virt_data_root.split('/')[-1]
        human_info = get_human_info.get_human_info(self.split)
        human_list = list(human_info.keys())

        if self.split == 'test':
            self.human_idx_name = {}
            for human_idx in range(len(human_list)):
                human = human_list[human_idx]
                self.human_idx_name[human] = human_idx

        for idx in range(len(human_list)):
            human = human_list[idx]

            data_root = os.path.join(cfg.virt_data_root, human)

            ann_file = os.path.join(cfg.virt_data_root, human, 'annots.npy')
            annots = np.load(ann_file, allow_pickle=True).item()

            self.cams[human] = annots['cams']

            num_cams = len(self.cams[human]['K'])

            if cfg.run_mode == 'train':
                if cfg.test_sample_cam:
                    if human in ['CoreView_313', 'CoreView_315']:
                        test_view = [i for i in range(num_cams) if i not in cfg.zju_313_315_sample_cam]
                    else:
                        test_view = [i for i in range(num_cams) if i not in cfg.zju_sample_cam]

            elif cfg.run_mode == 'test':
                if cfg.test_sample_cam:
                    if human in ['CoreView_313', 'CoreView_315']:
                        test_view = cfg.zju_313_315_sample_cam
                    else:
                        test_view = cfg.zju_sample_cam
                else:
                    test_view = [i for i in range(num_cams) if
                                 i not in cfg.test_input_view]

            if len(test_view) == 0:
                test_view = [0]


            i = 0
            i = i + human_info[human]['begin_i']
            i_intv = human_info[human]['i_intv']
            ni = human_info[human]['ni']

            end = i + ni
            num_frame = cfg.time_steps
            interval = int((ni-1) / num_frame)
            select_frame = np.arange(i+1, end, interval, dtype=int)[:num_frame]

            if cfg.run_mode == 'test':
                all_frames = range(i, i+ni, i_intv)
                all_frames = [*all_frames]
                test_frames = list(set(all_frames) - set(select_frame))
                test_frames_annot = [annots['ims'][f] for f in test_frames]
                ims = np.array([np.array(ims_data['ims'])[test_view] for ims_data in test_frames_annot]).ravel()
                cam_inds = np.array([np.arange(len(ims_data['ims']))[test_view] for ims_data in test_frames_annot]).ravel()
            else:
                ims = np.array([
                    np.array(ims_data['ims'])[test_view]
                    for ims_data in annots['ims'][i:i + ni][::i_intv]
                ]).ravel()

                cam_inds = np.array([
                    np.arange(len(ims_data['ims']))[test_view]
                    for ims_data in annots['ims'][i:i + ni][::i_intv]
                ]).ravel()

            start_idx = len(self.ims)
            length = len(ims)
            self.ims.extend(ims)
            self.cam_inds.extend(cam_inds)

            if human in ['CoreView_313', 'CoreView_315']:

                self.ims[start_idx:start_idx + length] = [
                    data_root + '/' + x.split('/')[0] + '/' +
                    x.split('/')[1].split('_')[4] + '.jpg' for x in
                    self.ims[start_idx:start_idx + length]]
            else:
                self.ims[start_idx:start_idx + length] = [
                    data_root + '/' + x for x in
                    self.ims[start_idx:start_idx + length]]

            self.start_end[human] = {}
            self.start_end[human]['start'] = int(
                self.ims[start_idx].split('/')[-1][:-4])
            self.start_end[human]['end'] = int(
                self.ims[start_idx + length - 1].split('/')[-1][:-4])
            self.start_end[human]['length'] = self.start_end[human]['end'] - \
                                              self.start_end[human]['start']

            self.start_end[human]['intv'] = human_info[human]['i_intv']

            self.select_frames[human] = select_frame

        self.nrays = cfg.N_rand
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
            [transforms.ColorJitter(brightness=(0.2, 2),
                                    contrast=(0.3, 2), saturation=(0.2, 2),
                                    hue=(-0.5, 0.5)), ]
        )

        return transforms.Compose(ops)

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

        data_name = cfg.virt_data_root.split('/')[-1]

        border = 5

        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)
        msk[(msk_dilate - msk_erode) == 1] = 100

        return msk

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

    def get_depth_map(self, human, cam, frame):

        depth_map_path = os.path.join('./data/zju_depthmap', human, cam, '{}.npy'.format(frame))
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
        params_path = os.path.join(cfg.virt_data_root, human, cfg.params,
                                   '{}.npy'.format(i))

        params = np.load(params_path, allow_pickle=True).item()


        Rh = params['Rh']
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = params['Th'].astype(np.float32)
        poses = params['poses'].reshape(-1, 3).astype(np.float32)


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

    def get_item(self, index):
        return self.__getitem__(index)

    def __getitem__(self, index):

        prob = np.random.randint(9000000)

        data_name = cfg.virt_data_root.split('/')[-1]
        img_path = self.ims[index]

        data_info = img_path.split('/')
        human = data_info[-3]
        camera = data_info[-2]
        frame = data_info[-1]

        img = imageio.imread(img_path)
        smpl_target = self.get_smpl_vertice(human, int(frame[:-4]))
        pbw = np.load(os.path.join(cfg.virt_data_root, human, cfg.lbs, 'bweights/{}.npy'.format(int(frame[:-4]))))
        pbw = pbw.astype(np.float32)

        jitter_aug = False
        if self.split == 'train' and cfg.jitter:
            if random.random() > cfg.aug_thre:
                img = Image.fromarray(img)
                torch.manual_seed(prob)
                img = self.jitter(img)
                img = np.array(img)
                jitter_aug = True

        img = img.astype(np.float32) / 255.

        msk = self.get_mask(index)

        cam_ind = self.cam_inds[index]
        K = np.array(self.cams[human]['K'][cam_ind])
        D = np.array(self.cams[human]['D'][cam_ind])

        if human in ['CoreView_313', 'CoreView_315']:
            cam_idx_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                            14, 15, 16, 17, 18, 21, 22]

        if cfg.camera_embedding:
            if human in ['CoreView_313', 'CoreView_315']:
                cam_embid = cam_idx_list[cam_ind]
            else:
                cam_embid = self.cam_inds[index]
        else:
            cam_embid = 0

        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)

        R = np.array(self.cams[human]['R'][cam_ind])
        # when generating the annots.py, 1000 is multiplied, so dividing back
        T = np.array(self.cams[human]['T'][cam_ind]) / 1000.

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)

        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

        if cfg.mask_bkgd:
            if cfg.white_bkgd:
                img[msk == 0] = 1
            else:
                img[msk == 0] = 0
        K[:2] = K[:2] * cfg.ratio

        target_K = K.copy()
        target_R = R.copy()
        target_T = T.copy()

        num_inputs = len(cfg['test_input_view'])

        lbs_root = os.path.join(cfg.virt_data_root, human, cfg.lbs)
        joints_t = np.load(os.path.join(lbs_root, 'joints.npy'))
        joints_t = joints_t.astype(np.float32)
        parents = np.load(os.path.join(lbs_root, 'parents.npy'))

        input_vizmaps = []
        input_imgs = []
        input_msks = []
        input_K = []
        input_R = []
        input_T = []
        smpl_vertices = []

        target_frame = frame[:-4]
        frame_index = int(target_frame)
        zfill = len(target_frame)

        start = self.start_end[human]['start']
        end = self.start_end[human]['end']
        length = self.start_end[human]['length']
        num_frame = cfg.time_steps
        interval = int(length / num_frame)
        select_frame = self.select_frames[human]

        # select observed frames randomly during training, and use fixed frames during test
        if cfg.run_mode == 'train':
            random_select = select_frame + np.random.randint(-int(interval/2), int(interval/2), num_frame)
            random_select[0] = max(start, random_select[0])
            random_select[-1] = min(end, random_select[-1])
        else:
            random_select = select_frame

        i = int(frame[:-4])

        feature, coord, out_sh, can_bounds, bounds, Rh, Th, center, rot, trans, tmp_smpl_vertices, poses = \
                                                        self.prepare_input(human, i)
        poses_target = poses.copy().reshape(-1)[3:] + 1e-3
        poses_target = torch.from_numpy(poses_target.astype(np.float32))
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
            current_frame = random_select[t]
            filename = str(current_frame).zfill(zfill) + '.jpg'

            if cfg.time_steps > 1:
                smpl_vertices.append(
                    self.get_smpl_vertice(human, current_frame))

            tmp_vizmaps = []
            tmp_imgs = []
            tmp_msks = []
            for i in range(num_inputs):
                idx = cam_ind
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

                input_img = imageio.imread(input_img_path)

                if self.split == 'train' and cfg.jitter:
                    if jitter_aug:
                        input_img = Image.fromarray(input_img)
                        torch.manual_seed(prob)
                        input_img = self.jitter(input_img)
                        input_img = np.array(input_img)

                input_img = input_img.astype(np.float32) / 255.

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
                            input_msk != 0)  # bool mask : foreground (True) background (False)

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

        rgb, ray_o, ray_d, near, far, coord_, mask_at_box = if_nerf_dutils.sample_ray_h36m(
            img, msk, K, R, T, can_bounds, self.nrays, self.split)
        acc = if_nerf_dutils.get_acc(coord_, msk)

        ret = {
            'smpl_vertice': smpl_vertices,
            'feature': feature,
            'coord': coord,
            'out_sh': out_sh,
            'rgb': rgb,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'acc': acc,
            'mask_at_box': mask_at_box,
            'vertices_target': smpl_target,
            'pbw': pbw,
            'lbs_weights': self.lbs_weights,
            'target_image': img
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
            'cam_ind': cam_ind,
            'frame_index': frame_index,
            'human_idx': human_idx,
            'input_imgs': input_imgs,
            'input_msks': input_msks,
            'input_vizmaps': input_vizmaps,
            'input_K': input_K,
            'input_R': input_R,
            'input_T': input_T,
            'target_K': target_K,
            'target_R': target_R,
            'target_T': target_T,
            'Rh_in': Rh_in,
            'Th_in': Th_in,
            'A_target': A_target,
            'A_in': A_in,
            'cam_embid': cam_embid,
            'pose_target': poses_target

        }
        ret.update(meta)

        return ret

    def get_length(self):
        return self.__len__()

    def __len__(self):
        return len(self.ims)

