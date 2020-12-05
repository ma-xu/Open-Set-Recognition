"""
Version2: includes centroids into model, and shares embedding layers.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import backbones.cifar as models
from Distance import Similarity, Distance

class Decorrelation(nn.Module):
    def __init__(self, channel,):
        super(Decorrelation, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,channel))
        self.bias = nn.Parameter(torch.zeros(1,channel))

    def forward(self, gap):
        _mean = gap.mean(dim=1,keepdim=True)
        _std = gap.std(dim=1,keepdim=True)
        y = (gap-_mean)/(_std+1e-5)
        y =y * self.weight+self.bias
        y = torch.sigmoid(y)
        return gap * y

class DFPNet(nn.Module):
    def __init__(self, backbone='ResNet18', num_classes=1000, embed_dim=512,
                 similarity="cosine"):
        super(DFPNet, self).__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone

        self.backbone = models.__dict__[backbone](num_classes=num_classes, backbone_fc=False)
        self.feat_dim = self.get_backbone_last_layer_out_channel()  # get the channel number of backbone output
        self.embed_dim = embed_dim

        self.embeddingLayer = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.feat_dim, self.feat_dim // 16, bias=False),
            nn.PReLU(),
            nn.Linear(self.feat_dim // 16, embed_dim, bias=False)
        )
        self.centroids = nn.Parameter(torch.randn(num_classes, embed_dim))
        self.similarity = similarity

    def get_backbone_last_layer_out_channel(self):
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
        gap = (F.adaptive_avg_pool2d(x, 1)).view(x.size(0), -1)
        embed_fea = F.normalize(self.embeddingLayer(gap), dim=1, p=2)
        centroids = F.normalize(self.centroids, dim=1, p=2)
        SIMI = Similarity()
        sim_fea2cen = getattr(SIMI, self.similarity)(embed_fea, centroids)

        return {
            "gap": gap,  # [n,self.feat_dim]
            "embed_fea": embed_fea,  # [n,embed_dim]
            "sim_fea2cen": sim_fea2cen  # [n,num_classes]
        }


def demo():
    x = torch.rand([3, 3, 32, 32])
    y = torch.rand([6, 3, 32, 32])
    net = DFPNet('ResNet18', num_classes=10, embed_dim=64)
    output = net(x)
    print(output)

demo()
