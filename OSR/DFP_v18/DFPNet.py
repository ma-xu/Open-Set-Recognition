"""
Version2: includes centroids into model, and shares embedding layers.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import backbones.cifar as models
from Distance import Similarity, Distance
from Generater import CGDestimator


class DFPNet(nn.Module):
    def __init__(self, backbone='ResNet18', num_classes=1000, embed_dim=None, distance='l2',
                 similarity="dotproduct", scaled=True, thresholds=None, norm_centroid=False):
        super(DFPNet, self).__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.norm_centroid = norm_centroid
        self.backbone = models.__dict__[backbone](num_classes=num_classes, backbone_fc=False)
        self.feat_dim = self.get_backbone_last_layer_out_channel()  # get the channel number of backbone output
        if embed_dim:
            self.embeddingLayer = nn.Sequential(
                nn.PReLU(),
                nn.Linear(self.feat_dim, self.feat_dim // 16),
                nn.PReLU(),
                nn.Linear(self.feat_dim // 16, embed_dim)
            )
            self.feat_dim = embed_dim
        self.centroids = nn.Parameter(torch.randn(num_classes, self.feat_dim))
        self.register_buffer("origin", torch.zeros([1, self.feat_dim]))

        self.distance = distance
        self.similarity = similarity
        self.scaled = scaled
        if thresholds is not None:
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

    def generator(self, x):
        b, c, w, h = x.size()
        x_bak = x.clone().detach()

        # mixup with the input data
        x_bak_noise = x_bak.unsqueeze(dim=0).expand([2, b, c, w, h])
        x_bak_noise = x_bak_noise.view([-1, c, w, h])
        x_bak_noise = x_bak_noise[torch.randperm(x_bak_noise.size()[0])]
        x_bak_noise = x_bak_noise[0:b]
        weight = x_bak_noise[b:]
        weight = weight[:,0,1,2] # random select the 0-th channel, 1/2-th pixel as weight
        weight = torch.sigmoid(weight)/2.0
        noise_mix_input = (1.0-weight)*x_bak + weight*x_bak_noise



        gen = self.backbone(noise_mix_input)
        return gen

    def forward(self, input):
        x = self.backbone(input)
        dis_gen2cen, dis_gen2ori, thresholds, amplified_thresholds, embed_gen = None, None, None, None, None
        gap = (F.adaptive_avg_pool2d(x, 1)).view(x.size(0), -1)
        if hasattr(self, 'thresholds'):
            thresholds = self.thresholds
            gen = self.generator(input)
            embed_gen = self.embeddingLayer(gen) if hasattr(self, 'embeddingLayer') else gen

        embed_fea = self.embeddingLayer(gap) if hasattr(self, 'embeddingLayer') else gap
        centroids = F.normalize(self.centroids, dim=1, p=2) if self.norm_centroid else self.centroids
        SIMI = Similarity(scaled=self.scaled)
        sim_fea2cen = getattr(SIMI, self.similarity)(embed_fea, centroids)
        DIST = Distance(scaled=self.scaled)
        dis_fea2cen = getattr(DIST, self.distance)(embed_fea, centroids)
        if hasattr(self, 'thresholds'):
            dis_gen2cen = getattr(DIST, self.distance)(embed_gen, centroids)
            dis_gen2ori = getattr(DIST, self.distance)(embed_gen, self.origin)

        return {
            "gap": gap,
            "embed_fea": embed_fea,
            "embed_gen": embed_gen,
            "sim_fea2cen": sim_fea2cen,
            "dis_fea2cen": dis_fea2cen,
            "dis_gen2cen": dis_gen2cen,
            "dis_gen2ori": dis_gen2ori,
            "thresholds": thresholds
        }


def demo():
    x = torch.rand([10, 3, 32, 32])
    y = torch.rand([6, 3, 32, 32])
    threshold = torch.rand([10])
    net = DFPNet('ResNet18', num_classes=10, embed_dim=64, thresholds=None)
    output = net(x)
    print(output["gap"].shape)
    print(output["embed_fea"].shape)
    print(output["sim_fea2cen"].shape)
    # print(output["dis_gen2cen"].shape)
    # print(output["dis_gen2ori"].shape)


# demo()
