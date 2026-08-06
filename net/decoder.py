import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, input_dim, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(4, input_dim)
        self.conv2 = nn.Conv2d(input_dim, input_dim, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(4, input_dim)

    def forward(self, x):
        residual = x
        out = F.gelu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.gelu(out + residual)


class Decoder(nn.Module):
    def __init__(self, input_dim=24, output_dim=1, hidden_dim=32):
        super(Decoder, self).__init__()

        self.multi_scale = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 5, padding=2, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 7, padding=3, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=3, dilation=3, bias=False),
        ])

        self.multi_scale_fusion = nn.Sequential(
            nn.Conv2d(hidden_dim // 2 * 4, hidden_dim, 1, bias=False),
            nn.GroupNorm(4, hidden_dim),
            nn.GELU()
        )

        self.stage1 = ResidualBlock(hidden_dim)

        self.channel_reduce1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim - 8, 3, padding=1, bias=False),
            nn.GroupNorm(4, hidden_dim - 8),
            nn.GELU()
        )

        self.skip_conv1 = nn.Conv2d(hidden_dim, hidden_dim - 8, 1, bias=False)

        self.skip_fusion1 = nn.Sequential(
            nn.Conv2d((hidden_dim - 8) * 2, hidden_dim - 8, 3, padding=1, bias=False),
            nn.GroupNorm(4, hidden_dim - 8),
            nn.GELU()
        )

        self.stage2 = ResidualBlock(hidden_dim - 8)

        self.channel_reduce2 = nn.Sequential(
            nn.Conv2d(hidden_dim - 8, hidden_dim - 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, hidden_dim - 16),
            nn.GELU()
        )

        self.skip_conv2 = nn.Conv2d(hidden_dim - 8, hidden_dim - 16, 1, bias=False)

        self.skip_fusion2 = nn.Sequential(
            nn.Conv2d((hidden_dim - 16) * 2, hidden_dim - 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, hidden_dim - 16),
            nn.GELU()
        )

        self.refine = ResidualBlock(hidden_dim - 16)

        self.output_conv = nn.Conv2d(hidden_dim - 16, output_dim, 1, bias=False)

    def forward(self, fused_feat):
        multi_scale_feats = []
        for conv in self.multi_scale:
            multi_scale_feats.append(conv(fused_feat))

        multi_scale_concat = torch.cat(multi_scale_feats, dim=1)
        fused_feat = self.multi_scale_fusion(multi_scale_concat) + fused_feat

        skip_32 = fused_feat

        fused_feat = self.stage1(fused_feat)

        skip_24 = self.skip_conv1(skip_32)

        fused_feat = self.channel_reduce1(fused_feat)

        fused_feat = torch.cat([fused_feat, skip_24], dim=1)
        fused_feat = self.skip_fusion1(fused_feat)

        skip_24_out = fused_feat

        fused_feat = self.stage2(fused_feat)

        skip_16 = self.skip_conv2(skip_24_out)

        fused_feat = self.channel_reduce2(fused_feat)

        fused_feat = torch.cat([fused_feat, skip_16], dim=1)
        fused_feat = self.skip_fusion2(fused_feat)

        fused_feat = self.refine(fused_feat)

        output = self.output_conv(fused_feat)

        return torch.sigmoid(output)
