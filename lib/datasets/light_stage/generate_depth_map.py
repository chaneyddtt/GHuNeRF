import torch
import numpy as np
import os
import cv2
import sys
sys.path.append('/home/lic/projects/GHuNeRF/')
import pickle
# from pytorch3d.renderer import FoVPerspectiveCameras, MeshRenderer, MeshRasterizer, RasterizationSettings, \
#     PerspectiveCameras
# from pytorch3d.structures import Meshes


data_path = './data/zju_mocap'
human_info = ['CoreView_313',
              'CoreView_315',
              'CoreView_377',
              'CoreView_386',
              'CoreView_387',
              'CoreView_390',
              'CoreView_392',
              'CoreView_393',
              'CoreView_394',
              ]

with open('./data/smplx/smpl/SMPL_NEUTRAL.pkl', 'rb') as f:
    smpl_info = pickle.load(f, encoding="latin1")
smpl_faces = torch.from_numpy(smpl_info['f'].astype(int))

save_path_root = './data/zju_depthmap'


def get_smpl_vertice(human, frame):
    vertices_path = os.path.join(data_path, human, 'vertices',
                                 '{}.npy'.format(frame))
    smpl_vertice = np.load(vertices_path).astype(np.float32)

    return smpl_vertice


def generate_depth_map_cuda_v2(debug=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl_faces = torch.from_numpy(smpl_info['f'].astype(int)).to(device)
    image_size = torch.tensor([512, 512])
    for humanid in human_info:
        data_root = os.path.join(data_path, humanid)
        if humanid in ['CoreView_313', 'CoreView_315']:
            cam_idx_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                            14, 15, 16, 17, 18, 21, 22]
        else:
            cam_idx_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                14, 15, 16, 17, 18, 19, 20, 21, 22]
        ann_file = os.path.join(data_path, humanid, 'annots.npy')
        annot = np.load(ann_file, allow_pickle=True).item()
        K = annot['cams']['K']
        R = annot['cams']['R']
        T = annot['cams']['T']

        ims = np.array([
            np.array(ims_data['ims'])
            for ims_data in annot['ims']
        ]).ravel()

        if humanid in ['CoreView_313', 'CoreView_315']:

            ims = np.array([
                data_root + '/' + x.split('/')[0] + '/' +
                x.split('/')[1].split('_')[4] + '.jpg' for x in
                ims]).reshape(-1, len(K))
        else:
            ims = np.array([data_root + '/' + x for x in ims]).reshape(-1, len(K))

        for idx in range(len(cam_idx_list)):
            cam = cam_idx_list[idx]

            if humanid in ['CoreView_313', 'CoreView_315']:
                save_path = os.path.join(save_path_root, humanid,
                                         'Camera (' + str(
                                             cam + 1) + ')')
            else:
                save_path = os.path.join(save_path_root, humanid,
                                         'Camera_B' + str(cam + 1))
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            K_idx = np.array(K[idx]) #* 0.5
            K_idx[:2] = K_idx[:2] * 0.5
            K_idx = torch.from_numpy(K_idx.astype(np.float32)).to(device)
            R_idx = torch.from_numpy(np.array(R[idx]).astype(np.float32)).to(device)
            R_idx_clone = R_idx[None, ...].clone()
            T_idx = torch.from_numpy((np.array(T[idx])/1000.).astype(np.float32)).to(device)
            T_idx_clone = T_idx[None, ...].clone()
            torch2colmap = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]],dtype=torch.float32).to(device)
            R_idx = (torch2colmap @ R_idx).T
            T_idx = torch2colmap @ T_idx

            K_idx = K_idx.unsqueeze(0)
            R_idx = R_idx.unsqueeze(0)
            T_idx = T_idx.squeeze(-1).unsqueeze(0)

            K_in = torch.zeros(size=(K_idx.shape[0], 4, 4)).to(device)
            K_in[:, :3, :3] = K_idx
            K_in[:, 2, -1] = 1
            K_in[:, -1, 2] = 1

            cameras = PerspectiveCameras(R=R_idx, T=T_idx, K=K_in, in_ndc=False, image_size=image_size.unsqueeze(0),
                                         device=device)
            # start = time.time()
            for f in range(len(ims)):
                img_path = ims[f][idx]
                frame_idx = img_path.split('/')[-1][:-4]
                smpl_vetices = get_smpl_vertice(humanid, int(frame_idx))
                smpl_vetices = torch.from_numpy(smpl_vetices).to(device)
                smpl_vetices = smpl_vetices.unsqueeze(0)
                raster_settings = RasterizationSettings(
                    image_size=(512, 512),
                    blur_radius=0.0,
                    faces_per_pixel=1
                )
                rasterizer = MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                )
                meshes = Meshes(verts=smpl_vetices, faces=smpl_faces.unsqueeze(0))
                depth = rasterizer(meshes).zbuf.squeeze(0).cpu().numpy()

                depth_save_path = os.path.join(save_path, frame_idx + '.npy')
                with open(depth_save_path, 'wb') as f:
                    np.save(f, depth)
                # depth_jh = np.load('/home/lic/Downloads/0001.npy')
                # depth = np.load(depth_save_path)
                # depth_jh[depth_jh==1e8]=0
                # depth[depth==-1]=0
                # depth_jh = depth_jh[..., None]
                # depth_diff = np.abs(depth_jh - depth)
                # depth_diff[depth_diff>0.01]=1
                # depth_diff[depth_diff < 0.01] = 0
                if debug:
                    vertice_rot = \
                        torch.matmul(R_idx_clone, smpl_vetices.permute(0, 2, 1))
                    vertice = vertice_rot + T_idx_clone
                    vertice_cam_depth = vertice[:, -1]
                    vertice = torch.matmul(K_idx, vertice).permute(0, 2, 1)
                    uv = vertice[:, :, :2] / vertice[:, :, 2:]

                    msk = np.zeros((512, 512))
                    uv_rescale = (uv.cpu().squeeze(0).numpy()).astype(np.int)
                    rows = uv_rescale[:, 1]
                    cols = uv_rescale[:, 0]
                    msk[rows, cols] = 1
                    vertice_cam_depthmap = depth[rows, cols]

                    # depth[depth==-1] = 0
                    # xy_depth = np.transpose(np.nonzero(depth[:, :, 0]))
                    # xy_depth = xy_depth[:, ::-1]
                    # z_depth = depth[depth!=0]
                    # uvz = np.concatenate([xy_depth, z_depth[:, None]], axis=1)
                    # uvz = torch.from_numpy(uvz.astype(np.float32)).to(device)
                    # points_unpro = cameras.unproject_points(uvz, world_coordinates=True)

                    # pcd = open3d.geometry.PointCloud()
                    # pcd.points = open3d.utility.Vector3dVector(points_unpro.cpu().numpy())
                    # pcd_v = open3d.geometry.PointCloud()
                    # pcd_v.points = open3d.utility.Vector3dVector(smpl_vetices[0].cpu().numpy())
                    # open3d.visualization.draw_geometries([pcd, pcd_v])

                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
                    img_masked = img * (depth!=-1).astype(np.float32)
                    depth[depth == 0] == 8.
                    depth = 1.0 / (depth + 1e-8)
                    depth = depth / (np.percentile(depth, 95) + 1e-6)
                    depth = np.clip(depth, 0, 1)

                    cv2.imshow('depth', depth)
                    cv2.imshow('img_msked', np.uint8(img_masked))
                    cv2.imshow('img', np.uint8(img))
                    cv2.imshow('msk', msk)
                    cv2.waitKey()


data_length = {'CoreView_313': 1060,
              'CoreView_315': 1400,
              'CoreView_377': 617,
              'CoreView_386': 646,
              'CoreView_387': 654,
              'CoreView_390': 999,
              'CoreView_392': 556,
              'CoreView_393': 658,
              'CoreView_394': 859}


# store the sampl vertices of a video sequence in one npy file to avoid reading once for each frame
def generate_smpl_file():
    for humanid in human_info:
        frame_length = data_length[humanid]
        smpl_vertices = []
        if humanid in ['CoreView_313', 'CoreView_315']:
            smpl_list = range(frame_length+1)[1::5]
        else:
            smpl_list = range(frame_length)[0::5]
        for f in smpl_list:
            vertices_path = os.path.join(data_path, humanid, 'vertices',
                                         '{}.npy'.format(f))
            smpl_vertice = np.load(vertices_path).astype(np.float32)
            smpl_vertices.append(smpl_vertice)
        smpl_vertices = np.stack(smpl_vertices, axis=0)
        smpl_save_path = os.path.join(data_path, humanid, 'vertices.npy')
        with open(smpl_save_path, 'wb') as f:
            np.save(f, smpl_vertices)


if __name__ == '__main__':
    # generate_depth_map_cuda_v2()
    generate_smpl_file()

