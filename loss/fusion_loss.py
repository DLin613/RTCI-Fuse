import os

from torch import nn

from loss.loss import Loss_Gradient, Loss_Intensity, Loss_SSIM, Loss_Contrast


class FusionLoss(nn.Module):
    def __init__(self, logs=True, config=None, batch_size=1, epochs=100):
        super(FusionLoss, self).__init__()
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.gamma = config['gamma']
        self.delta = config['delta']

        self.logs = logs
        self.batch_size = batch_size
        self.epochs = epochs

        self.loss_ssim = Loss_SSIM()
        self.loss_grad = Loss_Gradient()
        self.loss_intensity = Loss_Intensity()
        self.loss_contrast = Loss_Contrast()

    def log(self, loss_grad, loss_ssim, loss_intensity, loss_contrast, start_time, epoch):
        avg_grad = loss_grad.item() / self.batch_size
        avg_ssim = loss_ssim.item() / self.batch_size
        avg_intensity = loss_intensity.item() / self.batch_size
        avg_contrast = loss_contrast.item() / self.batch_size

        if self.alpha == 0:
            avg_grad = 0
        if self.beta == 0:
            avg_ssim = 0
        if self.gamma == 0:
            avg_intensity = 0
        if self.delta == 0:
            avg_contrast = 0

        if self.logs:
            os.makedirs('./detail_loss', exist_ok=True)
            with open(f'./detail_loss/{start_time}_loss_detail.txt', 'a') as f:
                f.write(
                    f"Epoch[{epoch}/{self.epochs}],avg loss detail[grad:{avg_grad:.6f},ssim:{avg_ssim:.6f},intensity:{avg_intensity:.6f},contrast:{avg_contrast:.6f}]\n")

    def forward(self, fused, ir, vis, start_time, epoch):
        ir_y = ir[:, 0:1]
        vis_y = vis[:, 0:1]

        loss_grad = self.loss_grad(fused, ir_y, vis_y)
        loss_ssim = self.loss_ssim(fused, ir_y, vis_y)
        loss_intensity = self.loss_intensity(fused, ir_y, vis_y)
        loss_contrast = self.loss_contrast(fused, ir_y, vis_y)

        total_loss = (self.alpha * loss_grad +
                      self.beta * loss_ssim +
                      self.gamma * loss_intensity +
                      self.delta * loss_contrast)

        self.log(loss_grad, loss_ssim, loss_intensity, loss_contrast, start_time, epoch)

        return total_loss
