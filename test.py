import glob
import os
import time

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from config.config import DEVICE
from net.network import Network
from utils.data_loader import TestDataLoader
from utils.utils import load_config


class Fusion:
    def __init__(self, config):
        self.data_path = config['data_path']
        self.data_set = config['data_set']
        self.save_path = config['save_path']
        self.model_path = config['model_path']
        self.config = config

    def __call__(self, *args, **kwargs):
        self.model = self.loadModel()
        test_ir_paths = sorted(glob.glob(os.path.join(self.data_path, self.data_set + '/ir', '*.*')))
        test_vis_paths = sorted(glob.glob(os.path.join(self.data_path, self.data_set + '/vis', '*.*')))
        train_data = TestDataLoader(test_ir_paths, test_vis_paths, mode='YCbCr')
        data_loader = DataLoader(train_data, shuffle=False, batch_size=1)
        data_loader = tqdm(data_loader)
        self.fusionProcess(self.model, data_loader)

    def loadModel(self):
        model = Network(self.config).to(DEVICE)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model

    def fusionProcess(self, model, data_loader):
        save_path = self.save_path + '/' + self.data_set
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with torch.no_grad():
            for ir, vis, fuse_image_name in data_loader:
                ir_y = ir[:, 0:1]
                vis_y = vis[:, 0:1]
                fused = model(ir_y, vis_y)
                fused_y = fused.clamp(0.0, 1.0)
                vis_cb_cr = vis[:, 1:3, :, :]
                fused_y_cb_cr = torch.cat([fused_y, vis_cb_cr], dim=1).squeeze(0).cpu()
                fused_y_cb_cr_pil = to_pil_image(fused_y_cb_cr, mode='YCbCr')
                fused_rgb_pil = fused_y_cb_cr_pil.convert('RGB')

                fused_save_path = save_path + '/' + fuse_image_name[0]
                fused_rgb_pil.save(fused_save_path)

if __name__ == '__main__':
    config = load_config(f'./config.json')

    data_path = 'F:/post/dataset'
    data_set = "MSRS"

    config['data_path'] = data_path
    model_name = f'RTCI-Fuse.pth'
    config['data_set'] = data_set
    config['model_path'] = f'./model/{model_name}'
    config['save_path'] = f'./RTCI-Fuse'
    fusion = Fusion(config=config)
    fusion()

