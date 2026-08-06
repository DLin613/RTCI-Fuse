import torch
from torch import nn

from modules.modules import CrossRepresentationEnhance, Projection


class CrossFusion(nn.Module):
    def __init__(self, input_dim=48, hidden_dim=24, output_dim=2):
        super(CrossFusion, self).__init__()
        self.weight = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_dim, 1, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, ir, vis):
        weight = self.weight(torch.cat([ir, vis], dim=1))
        w_ir = weight[:, 0:1, :, :]
        w_vis = weight[:, 1:2, :, :]
        fused = w_ir * ir + w_vis * vis
        return fused


class Fusion(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=32, window_size=8, dropout=0.1, config=None):
        super(Fusion, self).__init__()
        self.config = config

        self.CRE = CrossRepresentationEnhance(input_dim, window_size)

        self.proj_ir = Projection(input_dim, hidden_dim)
        self.proj_vis = Projection(input_dim, hidden_dim)

        self.cross_fusion = CrossFusion(hidden_dim * 2, hidden_dim // 2)

        self.enhanced_fusion = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(4, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(4, hidden_dim),
            nn.GELU()
        )

    def forward(self, cnn_ir, cnn_vis, gcn_ir, gcn_vis):
        enhanced_ir, enhanced_vis = self.CRE(cnn_ir, cnn_vis, gcn_ir, gcn_vis)
        feat_ir = self.proj_ir(enhanced_ir)
        feat_vis = self.proj_vis(enhanced_vis)
        cross_fused = self.cross_fusion(feat_ir, feat_vis)
        fused_feat = self.enhanced_fusion(cross_fused)

        return fused_feat
