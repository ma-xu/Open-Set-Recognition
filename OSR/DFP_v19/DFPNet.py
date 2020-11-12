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
        # x_bak = x.clone().detach()
        #
        # # mixup with the input data
        # x_bak_noise = x_bak.unsqueeze(dim=0).expand([2, b, c, w, h])
        # x_bak_noise = x_bak_noise.reshape([-1, c, w, h])
        # x_bak_noise = x_bak_noise[torch.randperm(x_bak_noise.size()[0])]
        # x_bak_noise_1 = x_bak_noise[0:b]
        # weight = x_bak_noise[b:]
        # weight = weight[:,0,1,2] # random select the 0-th channel, 1/2-th pixel as weight
        # weight = torch.sigmoid(weight)/2.0
        # weight = weight.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        # noise_mix_input = (1.0-weight)*x_bak + weight*x_bak_noise_1
        # gen = self.backbone(noise_mix_input)

        # # mixup with batch gussian distribution
        # gaussian_noise = torch.randn(b, c, w, h).to(x.device)
        # gaussian_noise = (gaussian_noise - gaussian_noise.mean(dim=0,keepdim=True))/(gaussian_noise.std(dim=0,keepdim=True))
        # batch_mean = x_bak.mean(dim=0,keepdim=True)
        # batch_std = x_bak.std(dim=0, keepdim=True)
        # gaussian_noise = gaussian_noise * batch_std + batch_mean
        # # noise_mix_gaussian = 0.7*x_bak + 0.3*gaussian_noise
        # noise_mix_gaussian = gaussian_noise
        # gen = self.backbone(noise_mix_gaussian)

        # channel_wise normalization (others reshape to 1 dim)
        gaussian_noise = torch.randn(b, c, w, h).to(x.device)
        noise_bak = gaussian_noise.permute(1, 0, 2, 3)
        noise_bak = noise_bak.reshape([c,-1])
        noise_mean = noise_bak.mean(dim=1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=0)
        noise_std = noise_bak.std(dim=1, keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=0)
        gaussian_noise.sub_(noise_mean).div_(noise_std)
        # gaussian_noise = 0.7*x_bak+0.15*gaussian_noise +0.15*x_bak_noise_1
        gen = self.backbone(gaussian_noise)

        gen = (F.adaptive_avg_pool2d(gen, 1)).view(x.size(0), -1)
        return gen

    def forward(self, input):
        x = self.backbone(input)
        gap = (F.adaptive_avg_pool2d(x, 1)).view(x.size(0), -1)
        embed_fea = self.embeddingLayer(gap) if hasattr(self, 'embeddingLayer') else gap
        gen = self.generator(input)
        embed_gen = self.embeddingLayer(gen) if hasattr(self, 'embeddingLayer') else gen

        centroids = F.normalize(self.centroids, dim=1, p=2) if self.norm_centroid else self.centroids
        SIMI = Similarity(scaled=self.scaled)
        sim_fea2cen = getattr(SIMI, self.similarity)(embed_fea, centroids)
        DIST = Distance(scaled=self.scaled)
        dis_fea2cen = getattr(DIST, self.distance)(embed_fea, centroids)
        dis_gen2cen = getattr(DIST, self.distance)(embed_gen, centroids)
        dis_gen2ori = getattr(DIST, self.distance)(embed_gen, self.origin)

        thresholds = None
        if hasattr(self, 'thresholds'):
            thresholds = self.thresholds

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
    x = torch.rand([3, 3, 32, 32])
    y = torch.rand([6, 3, 32, 32])
    threshold = torch.rand([10])
    net = DFPNet('ResNet18', num_classes=10, embed_dim=64, thresholds=None)
    output = net(x)
    print(output["gap"].shape)
    print(output["embed_fea"].shape)
    print(output["sim_fea2cen"].shape)
    print(output["dis_gen2cen"].shape)
    print(output["dis_gen2ori"].shape)


# demo()
