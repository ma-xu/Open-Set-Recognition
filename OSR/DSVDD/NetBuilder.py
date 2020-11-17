"""
DSVDD
Ruff, Lukas, et al. "Deep one-class classification." International conference on machine learning. 2018.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import backbones.cifar as models
from Distance import Distance


class NetBuilder(nn.Module):
    def __init__(self, backbone='LeNetNobias', embed_dim=2):
        super(NetBuilder, self).__init__()
        self.R = torch.Tensor(0)
        self.backbone_name = backbone
        self.backbone = models.__dict__[backbone](backbone_fc=False)
        self.backbone_dim = self._get_backbone_channel()
        self.embed_dim=embed_dim
        self.embeddingLayer = nn.Sequential(
            # DSVDD requires 3 conditions:
            # 1) the centroid can not be learnable
            # 2) the activation function can not be bounded, at least ReLU.
            # 3) all learnable layers can not includes bias.
            nn.PReLU(),
            nn.Linear(self.backbone_dim, self.embed_dim,bias=False)
        )
        self.register_buffer("centroid",torch.zeros([self.embed_dim]))

    def _init_centroid(self, centroid, eps=0.1):
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        centroid[(abs(centroid) < eps) & (centroid < 0)] = -eps
        centroid[(abs(centroid) < eps) & (centroid > 0)] = eps
        self.centroid = centroid.to(self.device)

    def _get_backbone_channel(self):
        if self.backbone_name.startswith("LeNet"):
            return 128 * 3 * 3
        last_layer = list(self.backbone.children())[-1]
        while (not isinstance(last_layer, nn.Conv2d)) and \
                (not isinstance(last_layer, nn.Linear)) and \
                (not isinstance(last_layer, nn.BatchNorm2d)):

            temp_layer = list(last_layer.children())[-1]
            if isinstance(temp_layer, nn.Sequential) and len(list(temp_layer.children())) == 0:
                temp_layer = list(last_layer.children())[-2]
            last_layer = temp_layer
        if isinstance(last_layer, nn.BatchNorm2d):
            return last_layer.num_features
        else:
            return last_layer.out_channels

    def forward(self, x):
        x = self.backbone(x)
        x = (F.adaptive_avg_pool2d(x, 1)).view(x.size(0), -1)
        x = self.embeddingLayer(x)
        DIST = Distance(scaled=True)
        dis = getattr(DIST, 'l2')(x, self.centroid)
        return {
            "embed_fea":x,
            "distance": dis,
            "radius": self.R
        }


def demo():
    x = torch.rand([2,1,28,28])
    net = NetBuilder()
    y = net(x)
    print(y)


# demo()
