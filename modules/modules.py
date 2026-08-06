import torch
from torch import nn
from components.components import CrossRepresentationInteractionModule
from torch_geometric.nn import LayerNorm, GCNConv, GATConv


class CrossRepresentationEnhance(nn.Module):
    def __init__(self, dim, window_size=8, dropout=0.1):
        super().__init__()
        self.cri_ir = CrossRepresentationInteractionModule(dim, window_size)
        self.cri_vis = CrossRepresentationInteractionModule(dim, window_size)

    def forward(self, cnn_ir, cnn_vis, gcn_ir, gcn_vis):
        enhanced_ir = self.cri_ir(cnn_ir, gcn_ir) + cnn_ir + gcn_ir
        enhanced_vis = self.cri_vis(cnn_vis, gcn_vis) + cnn_vis + gcn_vis

        return enhanced_ir, enhanced_vis


class GCNModule(nn.Module):
    def __init__(self, in_dim, hidden_dim, att_dim, out_dim):
        super(GCNModule, self).__init__()
        self.GConv1 = GCNConv(in_dim, hidden_dim)
        self.ln1 = LayerNorm(hidden_dim)
        self.at1 = nn.Mish()
        self.GATConv = GATConv(hidden_dim, att_dim, heads=3)
        self.ln2 = LayerNorm(att_dim * 3)
        self.at2 = nn.Mish()
        self.dp = nn.Dropout(0.2)
        self.GConv2 = GCNConv(att_dim * 3, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.GConv1(x, edge_index)
        x = self.ln1(x)
        x = self.at1(x)
        out = x

        x = self.GATConv(x, edge_index)

        x = self.ln2(x)
        x = self.at2(x)
        x = self.dp(x)
        final_out = self.GConv2(x, edge_index)

        return out, final_out


class Projection(nn.Module):
    def __init__(self, input_dim=24, output_dim=32):
        super(Projection, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, 1, bias=False),
            nn.GroupNorm(4, output_dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.layer(x)
