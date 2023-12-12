import matplotlib.pyplot as plt
import numpy as np
from lib.config import cfg

class Visualizer:
    def visualize(self, output, batch):
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()
        print('mse: {}'.format(np.mean((rgb_pred - rgb_gt) ** 2)))

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)


        if cfg.white_bkgd:
            img_pred = np.ones((H, W, 3))
            img_gt = np.ones((H, W, 3))
        else:
            img_pred = np.zeros((H, W, 3))
            img_gt = np.zeros((H, W, 3))

        img_pred[mask_at_box] = rgb_pred
        img_gt[mask_at_box] = rgb_gt

        _, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img_pred)
        ax2.imshow(img_gt)
        plt.show()
