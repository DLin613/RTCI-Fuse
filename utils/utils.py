import json

import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from torch_geometric.data import Data
from config.config import DEVICE


def sobel(image):
    # Sobel filters for x and y gradients
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
        image.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
        image.device)
    grad_x = F.conv2d(image, sobel_x, padding=1)
    grad_y = F.conv2d(image, sobel_y, padding=1)
    gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
    return gradient_magnitude


def SlicProcesses(img, fea_img, scale=2.0, sigma=5, compactness=10):
    _, c, h, w = img.shape
    fea_img = fea_img.clone().detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    init_segments = int(max(h, w) * scale)
    edge_list = []
    edge_img = sobel(img)

    edge_fea_list = edge_img.clone().detach().cpu().squeeze(0).permute(1, 2, 0).numpy()

    segments = slic(fea_img, n_segments=init_segments, sigma=sigma, compactness=compactness)
    superpixel_num = segments.max() + 1  # slic划分出的超像素个数
    segments_flatten = np.reshape(segments, [-1])

    fea_matrix = np.zeros([superpixel_num, c], dtype=np.float32)  # 初始化gcn运算特征矩阵 [num, channels]
    trans_matrix = torch.zeros([h * w, superpixel_num], dtype=torch.float32).to(
        DEVICE)  # 转换矩阵，将其特征提取后的转换到源图像上对应 [h*w, num]
    flatten_edge_img = np.reshape(edge_fea_list, [-1, c])

    # 计算每个超像素的中心位置
    superpixel_centers = np.zeros([superpixel_num, 2], dtype=np.float32)
    for i in range(superpixel_num):
        idx = np.where(segments_flatten == i)[0]
        if len(idx) == 0:
            continue
        rows = idx // w
        cols = idx % w
        superpixel_centers[i] = [np.mean(rows), np.mean(cols)]

    # 构建超像素特征矩阵和软分配转换矩阵
    pixel_coords = np.zeros([h * w, 2], dtype=np.float32)
    for idx in range(h * w):
        pixel_coords[idx] = [idx // w, idx % w]

    # 构建超像素特征矩阵
    sigma_soft = 5.0  # 高斯核的sigma，控制平滑程度

    for i in range(superpixel_num):
        idx = np.where(segments_flatten == i)[0]
        if len(idx) == 0:
            continue
        edge_img = flatten_edge_img[idx]
        edge_fea = np.sum(edge_img, 0)

        fea_matrix[i] = edge_fea / (len(idx) + 1e-8)  # 归一化

        # 软分配：基于距离的高斯加权
        pixel_positions = pixel_coords[idx]
        center = superpixel_centers[i]
        distances = np.sqrt(np.sum((pixel_positions - center) ** 2, axis=1))
        weights = np.exp(-(distances ** 2) / (2 * sigma_soft ** 2))
        weights = weights / (weights.sum() + 1e-8)  # 归一化

        trans_matrix[idx, i] = torch.from_numpy(weights).to(DEVICE)

    segments_ids = np.unique(segments)
    vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
    vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

    for i in range(bneighbors.shape[1]):
        node1 = bneighbors[0, i]
        node2 = bneighbors[1, i]
        idx1 = np.where(segments_ids == node1)[0][0]
        idx2 = np.where(segments_ids == node2)[0][0]
        # 构建邻接矩阵
        edge_list.append(idx1)
        edge_list.append(idx2)

    # Add self loops
    for i in range(len(segments_ids)):
        edge_list.append(i)
        edge_list.append(i)

    fea_matrix = torch.from_numpy(fea_matrix).to(DEVICE)

    edge_index = torch.tensor(edge_list).view(-1, 2).to(DEVICE)

    data = Data(x=fea_matrix, edge_index=edge_index.t())
    return data, trans_matrix


def getImage(image_path, convert="L"):
    image = Image.open(image_path).convert(convert)
    return image


def load_config(config_path='./config.json'):
    with open(config_path, 'r', encoding='utf-8') as f:
        configs = json.load(f)
    return configs
