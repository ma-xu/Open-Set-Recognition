"""
Version2: includes centroids into model, and shares embedding layers.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import backbones.cifar as models
from Distance import Distance
from Generater import generater_gap


class DFPNet(nn.Module):
    def __init__(self, backbone='ResNet18', num_classes=1000, embed_dim=None, distance='cosine', scaled=True,
                 thresholds=None):
        super(DFPNet, self).__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.backbone = models.__dict__[backbone](num_classes=num_classes, backbone_fc=False)
        self.feat_dim = self.get_backbone_last_layer_out_channel()  # get the channel number of backbone output
        if embed_dim:
            self.embeddingLayer = nn.Sequential(
                nn.PReLU(),
                nn.Linear(self.feat_dim, embed_dim)
            )
            self.feat_dim = embed_dim
        self.centroids = nn.Parameter(torch.randn(num_classes, self.feat_dim))
        self.register_buffer("original", torch.zeros([1, self.feat_dim]))
        self.classifier = nn.Linear(self.feat_dim, self.num_classes + 1)

        self.distance = distance
        assert self.distance in ['l1', 'l2', 'cosine']
        self.scaled = scaled
        self.register_buffer("thresholds", thresholds)

    def get_backbone_last_layer_out_channel(self):
        if self.backbone_name == "LeNetPlus":
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
        gap = (F.adaptive_avg_pool2d(x, 1)).view(x.size(0), -1)
        gap_gen = generater_gap(gap)
        embed_fea = self.embeddingLayer(gap) if hasattr(self, 'embeddingLayer') else gap
        logits = self.classifier(embed_fea)
        # calculate distance.
        cat_centroids = torch.cat([self.centroids, self.original], dim=0)
        DIST = Distance(scaled=self.scaled)
        dist_fea2cen = getattr(DIST, self.distance)(embed_fea, cat_centroids)

        embed_gen = self.embeddingLayer(gap_gen) if hasattr(self, 'embeddingLayer') else gap_gen
        dist_gen2cen = getattr(DIST, self.distance)(embed_gen, cat_centroids)

        return {
            "gap": x,
            "logits": logits,
            "embed_fea": embed_fea,
            "dist_fea2cen": dist_fea2cen,
            "dist_gen2cen": dist_gen2cen
        }


def demo():
    x = torch.rand([10, 3, 32, 32])
    y = torch.rand([6, 3, 32, 32])
    net = DFPNet('ResNet18', num_classes=10, embed_dim=64, thresholds=torch.rand(11))
    output = net(x,y)
    print(output["logits"].shape)
    print(output["embed_fea"].shape)
    print(output["dist_fea2cen"].shape)
    print(output["dist_gen2cen"].shape)

# demo()
