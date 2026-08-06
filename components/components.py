import torch
from torch import nn

class SimAM(nn.Module):
    def __init__(self, lamda=1e-5):
        super().__init__()
        self.lamda = lamda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w - 1
        mean = torch.mean(x, dim=[-2, -1], keepdim=True)
        var = torch.sum(torch.pow((x - mean), 2), dim=[-2, -1], keepdim=True) / n
        e_t = torch.pow((x - mean), 2) / (4 * (var + self.lamda)) + 0.5
        out = self.sigmoid(e_t) * x
        return out


class LSKblock(nn.Module):
    def __init__(self, dim):
        """
        LSKblock 初始化函数
        Args:
            dim (int): 输入通道的数量。
        """
        super().__init__()
        # 第一个卷积层，使用 5x5 的卷积核，通道数与输入一致，采用深度可分离卷积（groups=dim）。
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # 第二个卷积层，使用 7x7 的卷积核，采用深度可分离卷积，并设置扩展卷积（dilation=3）和较大的填充（padding=9）来扩大感受野。
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        # 第三个卷积层，使用 1x1 的卷积核，将通道数减少一半（dim -> dim//2）。
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        # 第四个卷积层，使用 1x1 的卷积核，将通道数减少一半（dim -> dim//2）。
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        # 用于通道间注意力机制的卷积层，接收 2 个通道的数据并输出 2 个通道，卷积核大小为 7x7。
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        # 最后一个卷积层，使用 1x1 的卷积核，将通道数恢复到原始输入的维度（dim//2 -> dim）。
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        """
        前向传播函数
        Args:
            x (Tensor): 输入张量，形状为 (batch_size, channels, height, width)。
        Returns:
            Tensor: 输出张量，形状与输入相同。
        """
        # 通过第一个卷积层，获取局部特征（5x5 卷积）。
        attn1 = self.conv0(x)
        # 通过第二个卷积层，获取更大范围的空间特征（7x7 扩展卷积）。
        attn2 = self.conv_spatial(attn1)
        # 通过第三个卷积层，减少通道数（1x1 卷积，dim -> dim//2）。
        attn1 = self.conv1(attn1)
        # 通过第四个卷积层，减少通道数（1x1 卷积，dim -> dim//2）。
        attn2 = self.conv2(attn2)
        # 将两个不同感受野的特征在通道维度上进行拼接，形成一个新的张量。
        attn = torch.cat([attn1, attn2], dim=1)
        # 对拼接后的特征图计算通道维度的平均值特征（avg_attn）。
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        # 对拼接后的特征图计算通道维度的最大值特征（max_attn）。
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        # 将平均值特征和最大值特征在通道维度上进行拼接。
        agg = torch.cat([avg_attn, max_attn], dim=1)
        # 通过注意力机制卷积层，生成两个通道的注意力权重。
        sig = self.conv_squeeze(agg).sigmoid()
        # 将两个不同感受野的特征分别乘以相应的注意力权重。
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        # 通过最后一个卷积层，将通道数恢复到原始输入的维度。
        attn = self.conv(attn)
        # 将原始输入与注意力加权后的特征相乘，得到增强后的输出。
        return x * attn


class WindowAttentionBase(nn.Module):
    def __init__(self, dim, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.scale = dim ** -0.5

    def window_attention(self, q, k, v):

        B, C, H, W = q.shape
        ws = self.window_size

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            q = nn.functional.pad(q, (0, pad_w, 0, pad_h))
            k = nn.functional.pad(k, (0, pad_w, 0, pad_h))
            v = nn.functional.pad(v, (0, pad_w, 0, pad_h))

        Hp, Wp = q.shape[2], q.shape[3]

        q_win = q.reshape(B, C, Hp // ws, ws, Wp // ws, ws).permute(2, 4, 0, 1, 3, 5)
        k_win = k.reshape(B, C, Hp // ws, ws, Wp // ws, ws).permute(2, 4, 0, 1, 3, 5)
        v_win = v.reshape(B, C, Hp // ws, ws, Wp // ws, ws).permute(2, 4, 0, 1, 3, 5)

        num_h_windows, num_w_windows = Hp // ws, Wp // ws
        num_windows = num_h_windows * num_w_windows
        q_win = q_win.reshape(num_windows * B, C, ws * ws)
        k_win = k_win.reshape(num_windows * B, C, ws * ws)
        v_win = v_win.reshape(num_windows * B, C, ws * ws)

        attn = torch.softmax((q_win.transpose(-2, -1) @ k_win) * self.scale, dim=-1)

        out_win = (v_win @ attn.transpose(-2, -1)).reshape(num_windows, B, C, ws, ws)
        out_win = out_win.permute(1, 2, 0, 3, 4).reshape(B, C, num_h_windows, num_w_windows, ws, ws)
        out = out_win.permute(0, 1, 2, 4, 3, 5).reshape(B, C, Hp, Wp)

        return out[:, :, :H, :W]


class CrossRepresentationInteractionModule(WindowAttentionBase):
    def __init__(self, dim, window_size=8, dropout=0.1):
        super().__init__(dim, window_size)


        self.q_fwd = nn.Conv2d(dim, dim, 1, bias=False)
        self.k_fwd = nn.Conv2d(dim, dim, 1, bias=False)
        self.v_fwd = nn.Conv2d(dim, dim, 1, bias=False)

        self.q_bwd = nn.Conv2d(dim, dim, 1, bias=False)
        self.k_bwd = nn.Conv2d(dim, dim, 1, bias=False)
        self.v_bwd = nn.Conv2d(dim, dim, 1, bias=False)

        self.proj_fwd = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.GroupNorm(4, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.proj_bwd = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.GroupNorm(4, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, cnn_feat, gcn_feat):
        out_fwd = self.proj_fwd(self.window_attention(
            self.q_fwd(cnn_feat), self.k_fwd(gcn_feat), self.v_fwd(gcn_feat)
        ))
        out_bwd = self.proj_bwd(self.window_attention(
            self.q_bwd(gcn_feat), self.k_bwd(cnn_feat), self.v_bwd(cnn_feat)
        ))

        return out_fwd + out_bwd
