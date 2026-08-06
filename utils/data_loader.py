import numpy as np
import torch
import h5py
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config.config import DEVICE, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH


class H5Dataset(Dataset):
    def __init__(self, h5file_path, split='train', val_ratio=0.1, train_sample_ratio=1.0):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['ir_patchs'].keys())
        h5f.close()
        
        total = len(self.keys)
        val_size = int(total * val_ratio)
        np.random.seed(42)
        val_indices = np.random.choice(total, val_size, replace=False)
        val_indices = set(val_indices)
        
        if split == 'train':
            self.keys = [k for i, k in enumerate(self.keys) if i not in val_indices]
            if train_sample_ratio < 1.0:
                sample_size = int(len(self.keys) * train_sample_ratio)
                np.random.seed(42)
                sample_indices = np.random.choice(len(self.keys), sample_size, replace=False)
                self.keys = [self.keys[i] for i in sample_indices]
        else:
            self.keys = [k for i, k in enumerate(self.keys) if i in val_indices]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        IR = np.array(h5f['ir_patchs'][key])
        VIS = np.array(h5f['vis_patchs'][key])
        h5f.close()
        return torch.Tensor(IR).to(DEVICE), torch.Tensor(VIS).to(DEVICE)



class IVIFDataLoader(Dataset):
    '''

    '''

    def __init__(self, ir_paths, vis_paths, mode='Normal'):
        '''
            ir_paths:红外图文件路径
            vis_paths:可见光图文件路径
            mode:图像读取方式 Normal(红外-GRAY,可见光-RGB) YCbCr(红外-Y通道)
        '''
        self.ir_paths = ir_paths
        self.vis_paths = vis_paths
        self.mode = mode
        self.trans = transforms.Compose([
            transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
        ])

    def __len__(self):
        if len(self.ir_paths) != len(self.vis_paths):
            raise ValueError("ir_paths and vis_paths must have the same length")
        return len(self.ir_paths)

    def __getitem__(self, idx):
        ir_path = self.ir_paths[idx]
        vis_path = self.vis_paths[idx]
        if self.mode == 'YCbCr':
            ir_image = Image.open(ir_path).convert('YCbCr')
            vis_image = Image.open(vis_path).convert('YCbCr')

        else:
            ir_image = Image.open(ir_path).convert('L')
            vis_image = Image.open(vis_path).convert('RGB')

        seed = torch.randint(0, 2 ** 32, (1,)).item()
        torch.manual_seed(seed)
        ir_image = self.trans(ir_image).to(DEVICE)
        torch.manual_seed(seed)
        vis_image = self.trans(vis_image).to(DEVICE)

        return ir_image, vis_image


class TestDataLoader(Dataset):
    """
    测试数据加载器
    """

    def __init__(self, ir_paths, vis_paths, mode='Normal'):
        self.ir_paths = ir_paths
        self.vis_paths = vis_paths
        self.mode = mode
        self.trans = transforms.Compose([
            # transforms.Resize((INPUT_IMAGE_HEIGHT * 2, INPUT_IMAGE_WIDTH * 2)),
            transforms.ToTensor()
        ])

    def __len__(self):
        if len(self.ir_paths) != len(self.vis_paths):
            raise ValueError("ir_paths and vis_paths must have the same length")
        return len(self.ir_paths)

    def __getitem__(self, idx):
        ir_path = self.ir_paths[idx]
        vis_path = self.vis_paths[idx]

        if self.mode == 'YCbCr':
            ir_image = Image.open(ir_path).convert('YCbCr')
            vis_image = Image.open(vis_path).convert('YCbCr')

        else:
            ir_image = Image.open(ir_path).convert('L')
            vis_image = Image.open(vis_path).convert('RGB')

        ir_image = self.trans(ir_image).to(DEVICE)
        vis_image = self.trans(vis_image).to(DEVICE)

        fuse_image_name = ir_path.split('\\')[-1]

        return ir_image, vis_image, fuse_image_name

