"""
Version2: includes centroids into model, and shares embedding layers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import backbones.cifar as models
from Distance import Distance


class DFPNet(nn.Module):
    def __init__(self, backbone='ResNet18', num_classes=1000,
                 backbone_fc=False, embed_dim=None, distance='l2', scaled=True):
        """

        :param backbone: the backbone architecture, default ResNet18
        :param num_classes: known classes
        :param backbone_fc: includes FC layers in backbone, default false.
        :param include_dist: if include the distance branch, default false to get traditional backbone architecture.
        :param embed_dim: if include the embedding layer and tell the embedding dimension.
        :param embed_reduction: for embedding reduction in SENet style. May deprecated.
        """
        super(DFPNet, self).__init__()
        assert backbone_fc == False  # drop out the classifier layer.
        # num_classes = num_classes would be useless if backbone_fc=False
        self.backbone = models.__dict__[backbone](num_classes=num_classes, backbone_fc=backbone_fc)
        self.feat_dim = self.get_backbone_last_layer_out_channel()  # get the channel number of backbone output
        if embed_dim:
            self.embeddingLayer = nn.Linear(self.feat_dim, embed_dim)
            self.feat_dim = embed_dim
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        self.centroids = nn.Parameter(torch.randn(num_classes, self.feat_dim))
        self.distance = distance
        assert self.distance in ['l1', 'l2', 'dotproduct']
        self.scaled = scaled

    def get_backbone_last_layer_out_channel(self):
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
        # TODO: extract more outputs from the backbone like FPN, but for intermediate weak-supervision.
        x = self.backbone(x)

        gap = F.adaptive_avg_pool2d(x, 1)
        gap = F.relu(gap.view(gap.size(0), -1), inplace=True)

        # if includes embedding layer.
        embed_fea = self.embeddingLayer(gap) if hasattr(self, 'embeddingLayer') else gap

        # processing the clssifier branch
        logits = self.classifier(embed_fea)

        # calculate distance.
        DIST = Distance(scaled=self.scaled)
        dist_fea2cen = getattr(DIST, self.distance)(embed_fea, self.centroids)  # [n, class_num]
        # dist_cen2cen = getattr(DIST, self.distance)(self.centroids,self.centroids)  # [class_num, class_num]

        return {
            "logits": logits,
            "embed_fea": embed_fea,
            "dist_fea2cen": dist_fea2cen,
            # "dist_cen2cen": dist_cen2cen
        }


def demo():
    x = torch.rand([1, 3, 32, 32])
    net = DFPNet('ResNet18', num_classes=100, embed_dim=64)
    output = net(x)
    print(output["logits"].shape)
    print(output["embed_fea"].shape)
    print(output["dist_fea2cen"].shape)
    # print(output["dist_cen2cen"].shape)


# demo()
