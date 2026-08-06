import torch
from pytorch_msssim import SSIM
from torch import nn

from utils.utils import sobel


class Loss_Intensity(nn.Module):
    def __init__(self):
        super(Loss_Intensity, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, fused, ir, vis):
        intensity_max = torch.max(ir, vis)
        intensity_min = torch.min(ir, vis)
        margin = 0.05
        loss_intensity = (torch.relu(fused - intensity_max - margin) +
                          torch.relu(intensity_min - margin - fused)).mean()
        return loss_intensity


class Loss_Gradient(nn.Module):
    def __init__(self):
        super(Loss_Gradient, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, fused, ir, vis):
        f_grad = sobel(fused)
        i_grad = sobel(ir)
        v_grad = sobel(vis)
        loss_grad = self.l1(f_grad, torch.max(i_grad, v_grad))
        return loss_grad


class Loss_Contrast(nn.Module):
    def __init__(self):
        super(Loss_Contrast, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, fused, ir, vis):
        fused_std = fused.std()
        target_std = torch.max(ir.std(), vis.std())
        loss_contrast = self.l1(fused_std, target_std)

        fused_max = fused.amax(dim=[2, 3])
        fused_min = fused.amin(dim=[2, 3])
        target_max = torch.max(ir.amax(dim=[2, 3]), vis.amax(dim=[2, 3]))
        target_min = torch.min(ir.amin(dim=[2, 3]), vis.amin(dim=[2, 3]))

        range_loss = (self.l1(fused_max, target_max) + self.l1(fused_min, target_min)) / 2

        return loss_contrast + range_loss


class Loss_SSIM(nn.Module):
    def __init__(self):
        super(Loss_SSIM, self).__init__()
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=1)

    def forward(self, fused, ir, vis):
        loss_ssim = 1 - self.ssim(fused, ir) + 1 - self.ssim(fused, vis)
        return loss_ssim
