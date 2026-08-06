import torch
import torch.nn as nn

from components.components import SimAM, LSKblock
from modules.modules import GCNModule
from utils.utils import SlicProcesses as img_processes


class CNNEncoder(nn.Module):
    def __init__(self, input_dim=1, output_dim=24, hidden_dim=16):
        super(CNNEncoder, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, 1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.GELU()
        )

        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.GELU()
        )

        self.normal = nn.Conv2d(hidden_dim * 2, hidden_dim, 1, bias=False)
        self.small =  nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1, bias=False)
        self.medium = nn.Conv2d(hidden_dim * 2, hidden_dim, 5, padding=2, bias=False)
        self.big = nn.Conv2d(hidden_dim * 2, hidden_dim, 7, padding=9, dilation=3, bias=False)

        self.feature_restore = nn.Sequential(
            nn.BatchNorm2d(hidden_dim * 4),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.GELU()
        )

        self.feature_projection = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, output_dim, 1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
            SimAM(),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, 1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.Mish(),
        )

    def forward(self, x):
        residual = self.residual(x)
        fea = self.stem(x)

        feat_normal = self.normal(fea)
        feat_small = self.small(fea)
        feat_medium = self.medium(fea)
        feat_big = self.big(fea)
        cat_feat = torch.cat([feat_small, feat_medium, feat_big, feat_normal], dim=1)

        feats = self.feature_restore(cat_feat)
        out = self.feature_projection(feats)
        out = self.final_conv(out)

        return out + residual


class GCNFusionGate(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(input_dim * 2, input_dim // 4, 1),
            nn.Sigmoid(),
            nn.Conv2d(input_dim // 4, 1, 1),
            nn.Sigmoid()
        )
        self.smooth = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 3, padding=1, groups=input_dim),
            nn.GroupNorm(4, input_dim),
            nn.GELU(),
            nn.Conv2d(input_dim, input_dim, 3, padding=1, groups=input_dim),
            nn.GroupNorm(4, input_dim),
        )

    def forward(self, shallow, deep):
        gate_weight = self.gate(torch.cat([shallow, deep], dim=1))

        fused = gate_weight * deep + (1 - gate_weight) * shallow

        fused = self.smooth(fused)

        return fused


class GCNEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=24, att_dim=48, output_dim=24, config=None):
        super(GCNEncoder, self).__init__()
        self.gcn_feat_weight = nn.Sequential(
            nn.Conv2d(1, 8, 1, bias=False),
            nn.GELU(),
            LSKblock(dim=8),
            nn.Conv2d(8, 1, 1, bias=False),
            nn.GELU()
        )

        self.gcn = GCNModule(input_dim, hidden_dim, att_dim, output_dim)
        self.scale = config['slic']['scale']
        self.sigma = config['slic']['sigma']
        self.compactness = config['slic']['compactness']
        if config is None:
            self.scale = 2.0
            self.sigma = 5
            self.compactness = 10

        self.gate = GCNFusionGate(output_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x_w = self.gcn_feat_weight(x)
        fea_x = (x_w * x + x) / 2
        gcn_list_x = []
        gcn_deep_x = []

        fea_list = torch.split(fea_x, 1, dim=0)
        list_x = torch.split(x, 1, dim=0)

        for i in range(b):
            gcn_data, trans_matrix = img_processes(list_x[i], fea_list[i], self.scale, self.sigma, self.compactness)
            out, gcn_result = self.gcn(gcn_data)
            gcn_result = torch.mm(trans_matrix, gcn_result)
            gcn_result = gcn_result.view(int(h), int(w), 24).permute(-1, 0, 1).unsqueeze(0)
            gcn_deep_x.append(gcn_result)
            out = torch.mm(trans_matrix, out)
            out = out.view(h, w, 24).permute(-1, 0, 1).unsqueeze(0)
            gcn_list_x.append(out)

        gcn_deep_feat = torch.cat(gcn_deep_x, dim=0)
        gcn_shallow = torch.cat(gcn_list_x, dim=0)

        gcn_feat = self.gate(gcn_shallow, gcn_deep_feat)

        return gcn_feat


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.gcn_ir = GCNEncoder(1, 24, 48, 24, config['gcn'])
        self.gcn_vis = GCNEncoder(1, 24, 48, 24, config['gcn'])

        self.cnn_ir = CNNEncoder(1, 24, 16)
        self.cnn_vis = CNNEncoder(1, 24, 16)

    def forward(self, ir, vis):
        gcn_ir = self.gcn_ir(ir)
        gcn_vis = self.gcn_vis(vis)

        cnn_ir = self.cnn_ir(ir)
        cnn_vis = self.cnn_vis(vis)

        return cnn_ir, cnn_vis, gcn_ir, gcn_vis
