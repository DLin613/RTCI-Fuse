from torch import nn
from net.encoder import Encoder
from net.decoder import Decoder
from net.fusion import Fusion

default_config = {
    "order": 1,
    "encoder": {
        "gcn": {
            "slic": {
                "scale": 2.0,
                "sigma": 5,
                "compactness": 10
            }
        }
    },
    "fusion": {
    },
    "loss": {
        "alpha": 2.5,
        "beta": 0.6,
        "gamma": 0.2,
        "delta": 0.3
    }
}


class Network(nn.Module):
    def __init__(self, config=None):
        super(Network, self).__init__()
        if config is None:
            config = default_config
        self.encoder = Encoder(config=config["encoder"])
        self.fusion = Fusion(config=config["fusion"])
        self.decoder = Decoder()

    def forward(self, ir, vis):
        cnn_ir, cnn_vis, gcn_ir, gcn_vis = self.encoder(ir, vis)
        fusion_feat = self.fusion(cnn_ir, cnn_vis, gcn_ir, gcn_vis)
        out = self.decoder(fusion_feat)
        return out
