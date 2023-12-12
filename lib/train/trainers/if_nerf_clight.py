import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import if_clight_renderer_bwv3
from lib.train import make_optimizer
import numpy as np
import cv2

class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.renderer = if_clight_renderer_bwv3.Renderer(self.net)

        self.img2mse = lambda x, y : torch.mean((x - y) ** 2)
        self.acc_crit = torch.nn.functional.smooth_l1_loss

    def forward(self, batch, val=False):
        ret = self.renderer.render(batch)
        image_stats = {}
        scalar_stats = {}
        mask = batch['mask_at_box']
        loss = 0
        if not val:
            img_loss = self.img2mse(ret['rgb_map'][mask], batch['rgb'][mask])
            scalar_stats.update({'img_loss': img_loss})
            loss += img_loss

            if 'rgb0' in ret:
                img_loss0 = self.img2mse(ret['rgb0'], batch['rgb'])
                scalar_stats.update({'img_loss0': img_loss0})
                loss += img_loss0
            scalar_stats.update({'loss': loss})

        else:
            rgb_pred = ret['rgb_map'][0].detach().cpu().numpy()
            rgb_gt = batch['rgb'][0].detach().cpu().numpy()
            mask_at_box = mask[0].detach().cpu().numpy()
            H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
            mask_at_box = mask_at_box.reshape(H, W)
            if cfg.white_bkgd:
                img_pred = np.ones((H, W, 3))
            else:
                img_pred = np.zeros((H, W, 3))
            img_pred[mask_at_box] = rgb_pred
            if cfg.white_bkgd:
                img_gt = np.ones((H, W, 3))
            else:
                img_gt = np.zeros((H, W, 3))
            img_gt[mask_at_box] = rgb_gt

            x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
            img_pred = img_pred[y:y + h, x:x + w]
            img_gt = img_gt[y:y + h, x:x + w]

            # image_stats.update({'img_pred': np.uint(img_pred[..., [2, 1, 0]] * 255)})
            # image_stats.update({'img_gt': np.uint(img_gt[..., [2, 1, 0]] * 255)})

            image_stats.update({'img_pred': img_pred})
            image_stats.update({'img_gt': img_gt})

        return ret, loss, scalar_stats, image_stats
