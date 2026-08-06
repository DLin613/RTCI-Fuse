<div align="center">

# RTCI-Fuse

Regional Topology Representation Learning and Cross-Representation Interaction for Infrared and Visible Image Fusion

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900?logo=nvidia&logoColor=white)
</div>

---

## 目录

- [目录结构](#目录结构)
- [环境依赖](#环境依赖)
- [项目启动主要配置说明](#项目启动主要配置说明)
  - [全局配置 config.json](#全局配置-configjson)
  - [训练脚本参数 train.py](#训练脚本参数-trainpy)
  - [测试脚本参数 test.py](#测试脚本参数-testpy)
- [测试数据集结构介绍](#测试数据集结构介绍)
  - [训练数据（H5 格式）](#训练数据h5-格式)
  - [测试数据（图像文件夹）](#测试数据图像文件夹)
- [快速启动](#快速启动)
- [训练日志与产物](#训练日志与产物)

---

## 目录结构

```
RTCI-Fuse/
├── components/              # 基础组件（SimAM、LSKblock、窗口注意力）
│   └── components.py
├── config/
│   └── config.py            # 设备与图像尺寸常量
├── detail_loss/             # 训练损失明细日志（自动生成）
├── h5/
│   └── MSRS_imgsize_64.h5   # 训练用 HDF5 数据集
├── loss/
│   ├── fusion_loss.py       # 融合损失加权组合
│   └── loss.py              # 梯度/强度/SSIM/对比度四个子损失
├── model/
│   └── RTCI-Fuse.pth        # 训练保存的权重文件
├── modules/
│   └── modules.py           # GCN 模块、CRE、Projection
├── net/
│   ├── encoder.py           # CNN/GCN 编码器
│   ├── fusion.py            # 跨模态融合模块
│   ├── decoder.py           # 多尺度残差解码器
│   └── network.py           # 整体网络封装
├── utils/
│   ├── data_loader.py       # H5 / 训练 / 测试 数据加载器
│   ├── early_stopping.py    # 早停策略
│   └── utils.py             # SLIC、Sobel、配置读取等工具
├── config.json              # 全局超参配置
├── train.py                 # 训练入口
└── test.py                  # 测试入口
```

---

## 环境依赖

`torch`
`torchvision`
`torch_geometric`
`h5py`
`numpy`
`pillow`
`scikit-image`
`pytorch_msssim`
`tqdm`
---

## 项目启动主要配置说明

### 全局配置 config.json

项目根目录下的 [`config.json`](file:///f:/post/code/RTCI-Fuse/config.json) 是网络结构与损失权重的核心配置文件：

```json
{
  "order": 1,
  "name": "normal",
  "image_size": 64,
  "encoder": {
    "gcn": {
      "slic": {
        "scale": 2.0,
        "sigma": 5,
        "compactness": 10
      }
    }
  },
  "fusion": {},
  "loss": {
    "alpha": 2.5,
    "beta": 0.6,
    "gamma": 0.2,
    "delta": 0.3
  }
}
```

| 字段 | 含义 | 说明 |
| --- | --- | --- |
| `order` | 实验序号 | 用于区分不同实验批次 |
| `name` | 实验名称 | 当前实验标识 |
| `image_size` | 训练 patch 尺寸 | 与 `h5` 数据集切片大小一致（64） |
| `encoder.gcn.slic.scale` | SLIC 超像素数量系数 | 实际超像素数 ≈ `max(h, w) * scale` |
| `encoder.gcn.slic.sigma` | SLIC 高斯平滑参数 | 控制分割边界平滑度 |
| `encoder.gcn.slic.compactness` | SLIC 紧凑度 | 平衡空间距离与色彩距离 |
| `loss.alpha` | 梯度损失权重 | 主导融合清晰度 |
| `loss.beta` | SSIM 损失权重 | 控制结构相似性 |
| `loss.gamma` | 强度损失权重 | 约束亮度范围 |
| `loss.delta` | 对比度损失权重 | 增强对比信息 |

### 训练脚本参数 train.py

[`train.py`](file:///f:/post/code/RTCI-Fuse/train.py) 的 `__main__` 入口集中了训练超参，可直接修改：

```python
config = load_config(f'./config.json')
train_data_path = './h5/MSRS_imgsize_64.h5'
epochs = 80
lr = 1e-4
batch_size = 16

model_name = f'RTCI-Fuse.pth'
config['data_path'] = train_data_path
config['epochs'] = epochs
config['batch_size'] = batch_size
config['model_name'] = model_name
config['save_path'] = f'./model/'
config['lr'] = lr
```

### 测试脚本参数 test.py

[`test.py`](file:///f:/post/code/RTCI-Fuse/test.py) 的 `__main__` 入口配置推理路径：

```python
config = load_config(f'./config.json')

data_path = 'F:/post/dataset'        # 测试数据集根目录
data_set = "MSRS"                    # 子数据集名称

config['data_path'] = data_path
model_name = f'RTCI-Fuse.pth'
config['data_set'] = data_set
config['model_path'] = f'./model/{model_name}'
config['save_path'] = f'./RTCI-Fuse'
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `data_path` | `F:/post/dataset` | 测试数据集根目录 |
| `data_set` | `MSRS` | 子数据集名称（与目录名一致） |
| `model_path` | `./model/RTCI-Fuse.pth` | 权重文件路径 |
| `save_path` | `./RTCI-Fuse` | 融合结果输出目录 |

---

## 测试数据集结构介绍

### 训练数据（H5 格式）

训练数据为 [`h5/MSRS_imgsize_64.h5`](file:///f:/post/code/RTCI-Fuse/h5/MSRS_imgsize_64.h5)，内部结构如下：

```
MSRS_imgsize_64.h5
├── /ir_patchs/        # 红外图像 patch 集合
│   ├── 0001           # key 为样本编号（字符串）
│   ├── 0002
│   └── ...
└── /vis_patchs/       # 可见光图像 patch 集合
    ├── 0001           # 与 ir_patchs 同名 key 一一对应
    ├── 0002
    └── ...
```

**说明**：

- 每个 key 对应一对 `(ir, vis)` patch，shape 通常为 `(64, 64)` 或 `(1, 64, 64)`。
- IR 与 VIS 通过 **相同的 key** 配对，加载数据时按 key 索引（见 [`H5Dataset`](file:///f:/post/code/RTCI-Fuse/utils/data_loader.py)）。

**下载地址**：

> H5 训练数据集下载：[MSRS_imgsize_64.h5](https://pan.baidu.com/s/10PQQa2IMvGhuBhLeIBUiBg?pwd=te22)  `提取码: te22`
>
> <!-- 请将上方链接替换为实际分享地址，例如百度网盘 / Google Drive / OneDrive 等 -->

### 测试数据（图像文件夹）

测试数据集采用目录约定，由 [`test.py`](file:///f:/post/code/RTCI-Fuse/test.py) 中的 `data_path` 与 `data_set` 组合定位：

```
F:/post/dataset/                   ← data_path
└── MSRS/                          ← data_set
    ├── ir/                        ← 红外图像目录
    │   ├── 0001.png
    │   ├── 0002.png
    │   └── ...
    └── vis/                       ← 可见光图像目录
        ├── 0001.png               ← 与 ir/ 同名图像配对
        ├── 0002.png
        └── ...
```

**目录约定**：

- `{data_path}/{data_set}/ir/` 存放红外图像。
- `{data_path}/{data_set}/vis/` 存放可见光图像。
- IR 与 VIS 图像通过 **同名文件** 配对，加载时使用 `sorted(glob(...))` 保证顺序一致。
- 支持 `png / jpg / jpeg / bmp` 等常见格式（`glob('*.*')` 匹配）。
- **两目录文件名必须一一对应**，否则 [`TestDataLoader`](file:///f:/post/code/RTCI-Fuse/utils/data_loader.py) 会抛出 `ValueError`。

---

## 快速启动

**1. 训练**

```bash
# 确保 h5/MSRS_imgsize_64.h5 已就位
python train.py
```

训练过程中：

- 终端实时输出每轮 `total loss / avg loss`；
- 检查点保存在 `./model/RTCI-Fuse_{epoch}.pth`；
- 最终最优权重保存为 `./model/RTCI-Fuse.pth`；
- 损失明细写入 `./detail_loss/{start_time}_loss_detail.txt`。

**2. 测试**

```bash
# 1) 修改 test.py 中的 data_path / data_set 指向你的数据集
# 2) 确保 ./model/RTCI-Fuse.pth 存在
python test.py
```

融合结果将保存至 `./RTCI-Fuse/{data_set}/` 目录，文件名与输入 IR 图像一致。

---

## 训练日志与产物

训练完成后会生成以下产物：

```
RTCI-Fuse/
├── model/
│   ├── RTCI-Fuse.pth              # 最终保存的权重（早停最优或最后一轮）
│   └── RTCI-Fuse_{epoch}.pth      # 中途检查点
├── detail_loss/
│   └── {YYYYMMDD_HH_MM_SS}_loss_detail.txt   # 每轮各项损失明细
└── RTCI-Fuse/                     # 测试输出（运行 test.py 后生成）
    └── MSRS/
        ├── 0001.png               # 融合结果
        └── ...
```

损失明细文件示例（每轮一行）：

```
Epoch[1/80],avg loss detail[grad:0.014089,ssim:0.107453,intensity:0.006627,contrast:0.010811]
Epoch[2/80],avg loss detail[grad:0.012377,ssim:0.102008,intensity:0.006691,contrast:0.013351]
```

---

<div align="center">

<sub>如果本项目对您的研究有帮助，欢迎 ⭐ Star 支持。</sub>

</div>
