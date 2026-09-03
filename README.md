<div align="center">

# RTCI-Fuse

Regional Topology Representation Learning and Cross-Representation Interaction for Infrared and Visible Image Fusion

[简体中文](./README_zh-CN.md) | **English**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900?logo=nvidia&logoColor=white)
</div>

---

## Table of Contents

- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)
- [Main Configuration](#main-configuration)
  - [Global Config config.json](#global-config-configjson)
  - [Training Script Parameters train.py](#training-script-parameters-trainpy)
  - [Testing Script Parameters test.py](#testing-script-parameters-testpy)
- [Dataset Structure](#dataset-structure)
  - [Training Data (H5 Format)](#training-data-h5-format)
  - [Test Data (Image Folders)](#test-data-image-folders)
- [Quick Start](#quick-start)
- [Training Logs and Artifacts](#training-logs-and-artifacts)

---

## Directory Structure

```
RTCI-Fuse/
├── components/              # Base components (SimAM, LSKblock, window attention)
│   └── components.py
├── config/
│   └── config.py            # Device and image size constants
├── detail_loss/             # Training loss detail logs (auto-generated)
├── h5/
│   └── MSRS_imgsize_64.h5   # HDF5 dataset for training
├── loss/
│   ├── fusion_loss.py       # Weighted fusion loss combination
│   └── loss.py              # Four sub-losses: gradient/intensity/SSIM/contrast
├── model/
│   └── RTCI-Fuse.pth        # Saved weight file
├── modules/
│   └── modules.py           # GCN module, CRE, Projection
├── net/
│   ├── encoder.py           # CNN/GCN encoder
│   ├── fusion.py            # Cross-modal fusion module
│   ├── decoder.py           # Multi-scale residual decoder
│   └── network.py           # Overall network wrapper
├── utils/
│   ├── data_loader.py       # H5 / training / test data loaders
│   ├── early_stopping.py    # Early stopping strategy
│   └── utils.py             # SLIC, Sobel, config reading utilities
├── config.json              # Global hyperparameter configuration
├── train.py                 # Training entry
└── test.py                  # Testing entry
```

---

## Dependencies

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

## Main Configuration

### Global Config config.json

The [`config.json`](file:///f:/post/code/RTCI-Fuse/config.json) in the project root directory is the core configuration file for network structure and loss weights:

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

| Field | Meaning | Description |
| --- | --- | --- |
| `order` | Experiment number | Used to distinguish different experiment batches |
| `name` | Experiment name | Current experiment identifier |
| `image_size` | Training patch size | Consistent with the slice size of the `h5` dataset (64) |
| `encoder.gcn.slic.scale` | SLIC superpixel count coefficient | Actual superpixel count ≈ `max(h, w) * scale` |
| `encoder.gcn.slic.sigma` | SLIC Gaussian smoothing parameter | Controls the smoothness of segmentation boundaries |
| `encoder.gcn.slic.compactness` | SLIC compactness | Balances spatial and color distances |
| `loss.alpha` | Gradient loss weight | Dominates fusion sharpness |
| `loss.beta` | SSIM loss weight | Controls structural similarity |
| `loss.gamma` | Intensity loss weight | Constrains brightness range |
| `loss.delta` | Contrast loss weight | Enhances contrast information |

### Training Script Parameters train.py

The `__main__` entry of [`train.py`](file:///f:/post/code/RTCI-Fuse/train.py) centralizes the training hyperparameters, which can be modified directly:

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

### Testing Script Parameters test.py

The `__main__` entry of [`test.py`](file:///f:/post/code/RTCI-Fuse/test.py) configures the inference paths:

```python
config = load_config(f'./config.json')

data_path = 'F:/post/dataset'        # Test dataset root directory
data_set = "MSRS"                    # Sub-dataset name

config['data_path'] = data_path
model_name = f'RTCI-Fuse.pth'
config['data_set'] = data_set
config['model_path'] = f'./model/{model_name}'
config['save_path'] = f'./RTCI-Fuse'
```

| Parameter | Default | Description |
| --- | --- | --- |
| `data_path` | `F:/post/dataset` | Test dataset root directory |
| `data_set` | `MSRS` | Sub-dataset name (must match the directory name) |
| `model_path` | `./model/RTCI-Fuse.pth` | Weight file path |
| `save_path` | `./RTCI-Fuse` | Fusion result output directory |

---

## Dataset Structure

### Training Data (H5 Format)

The training data is [`h5/MSRS_imgsize_64.h5`](file:///f:/post/code/RTCI-Fuse/h5/MSRS_imgsize_64.h5), with the following internal structure:

```
MSRS_imgsize_64.h5
├── /ir_patchs/        # Infrared image patch collection
│   ├── 0001           # key is the sample number (string)
│   ├── 0002
│   └── ...
└── /vis_patchs/       # Visible image patch collection
    ├── 0001           # One-to-one correspondence with same-named key in ir_patchs
    ├── 0002
    └── ...
```

**Notes**:

- Each key corresponds to a pair of `(ir, vis)` patches, with shape typically `(64, 64)` or `(1, 64, 64)`.
- IR and VIS are paired via **the same key**, and indexed by key when loading data (see [`H5Dataset`](file:///f:/post/code/RTCI-Fuse/utils/data_loader.py)).

**Download**:

> H5 training dataset download: [MSRS_imgsize_64.h5](https://pan.baidu.com/s/10PQQa2IMvGhuBhLeIBUiBg?pwd=te22)  `extraction code: te22`
>
> <!-- Please replace the above link with the actual share address, e.g. Baidu Netdisk / Google Drive / OneDrive etc. -->

### Test Data (Image Folders)

The test dataset uses a directory convention, located by combining `data_path` and `data_set` in [`test.py`](file:///f:/post/code/RTCI-Fuse/test.py):

```
F:/post/dataset/                   ← data_path
└── MSRS/                          ← data_set
    ├── ir/                        ← infrared image directory
    │   ├── 0001.png
    │   ├── 0002.png
    │   └── ...
    └── vis/                       ← visible image directory
        ├── 0001.png               ← paired with same-named image in ir/
        ├── 0002.png
        └── ...
```

**Directory Convention**:

- `{data_path}/{data_set}/ir/` stores infrared images.
- `{data_path}/{data_set}/vis/` stores visible images.
- IR and VIS images are paired via **same filename**, using `sorted(glob(...))` during loading to ensure consistent ordering.
- Supports common formats such as `png / jpg / jpeg / bmp` (matched by `glob('*.*')`).
- **Filenames in both directories must correspond one-to-one**, otherwise [`TestDataLoader`](file:///f:/post/code/RTCI-Fuse/utils/data_loader.py) will raise a `ValueError`.

---

## Quick Start

**1. Training**

```bash
# Ensure h5/MSRS_imgsize_64.h5 is in place
python train.py
```

During training:

- The terminal outputs `total loss / avg loss` for each epoch in real time;
- Checkpoints are saved at `./model/RTCI-Fuse_{epoch}.pth`;
- The final optimal weight is saved as `./model/RTCI-Fuse.pth`;
- Loss details are written to `./detail_loss/{start_time}_loss_detail.txt`.

**2. Testing**

```bash
# 1) Modify data_path / data_set in test.py to point to your dataset
# 2) Ensure ./model/RTCI-Fuse.pth exists
python test.py
```

Fusion results will be saved to the `./RTCI-Fuse/{data_set}/` directory, with filenames identical to the input IR images.

---

## Training Logs and Artifacts

The following artifacts are generated after training:

```
RTCI-Fuse/
├── model/
│   ├── RTCI-Fuse.pth              # Final saved weight (early-stopping best or last epoch)
│   └── RTCI-Fuse_{epoch}.pth      # Intermediate checkpoints
├── detail_loss/
│   └── {YYYYMMDD_HH_MM_SS}_loss_detail.txt   # Per-epoch loss details
└── RTCI-Fuse/                     # Test output (generated after running test.py)
    └── MSRS/
        ├── 0001.png               # Fusion result
        └── ...
```

Loss detail file example (one line per epoch):

```
Epoch[1/80],avg loss detail[grad:0.014089,ssim:0.107453,intensity:0.006627,contrast:0.010811]
Epoch[2/80],avg loss detail[grad:0.012377,ssim:0.102008,intensity:0.006691,contrast:0.013351]
```

---

<div align="center">

<sub>If this project helps your research, please ⭐ Star to support it.</sub>

</div>
