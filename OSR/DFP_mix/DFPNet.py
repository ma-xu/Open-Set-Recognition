"""
Version2: includes centroids into model, and shares embedding layers.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import backbones.cifar as models
from Distance import Distance


class DFPNet(nn.Module):
    def __init__(self, backbone='ResNet18', num_classes=1000,
                 backbone_fc=False, embed_dim=None, distance='cosine', scaled=True, cosine_weight=1.0, thresholds=None):
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
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.backbone = models.__dict__[backbone](num_classes=num_classes, backbone_fc=backbone_fc)
        self.feat_dim = self.get_backbone_last_layer_out_channel()  # get the channel number of backbone output
        if embed_dim:
            self.embeddingLayer = nn.Linear(self.feat_dim, embed_dim)
            self.feat_dim = embed_dim
        # self.classifier = nn.Linear(self.feat_dim, num_classes)
        # We add 1 centroid for the unknown class, which is like a placeholder.
        self.centroids = nn.Parameter(torch.randn(num_classes + 1, self.feat_dim))
        self.init_parameters()

        self.distance = distance
        assert self.distance in ['l1', 'l2', 'cosine']
        self.scaled = scaled
        self.cosine_weight = cosine_weight
        self.register_buffer("thresholds", thresholds)

    def init_parameters(self):
        # centroids = self.centroids-self.centroids.mean(dim=0,keepdim=True)
        # # the greater std is, seems can achieve by
        # centroids = centroids/(0.5*centroids.std(dim=0,keepdim=True))
        #
        # self.centroids = nn.Parameter(centroids)
        nn.init.normal_(self.centroids,mean=0., std=2.)
        print(f"Initilized Centroids: \n {self.centroids}")
        print(f"Initilized Centroids STD: \n {torch.std(self.centroids,dim=0)}")
        print(f"Initilized Centroids MEAN: \n {torch.mean(self.centroids, dim=0)}")
        # nn.init.normal_(self.centroids)


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

    def generat_rand_feature(self, gap, sampler=6):
        # generate a tensor with same shape as gap.
        n, c = gap.shape
        pool = gap.repeat(sampler, 1)  # repeat examples 3 times [n*sampler, c]
        pool_random = pool[torch.randperm(pool.size()[0])]
        pool_random = pool_random.view(sampler, n, c)
        pool_random = pool_random.mean(dim=0, keepdim=False)
        return pool_random

    def forward(self, x):
        # TODO: extract more outputs from the backbone like FPN, but for intermediate weak-supervision.
        x = self.backbone(x)
        dist_gen2cen = None

        gap = (F.adaptive_avg_pool2d(x, 1)).view(x.size(0), -1)
        if self.thresholds is not None:
            generate = self.generat_rand_feature(gap.clone()) # !!!need clone function, or gradient problem, shit
            generate_fea = F.relu(generate, inplace=True)
            # if includes embedding layer.
            generate_fea = self.embeddingLayer(generate_fea) if hasattr(self, 'embeddingLayer') else generate_fea
        gap = F.relu(gap, inplace=True)

        # if includes embedding layer.
        embed_fea = self.embeddingLayer(gap) if hasattr(self, 'embeddingLayer') else gap

        # calculate distance.
        DIST = Distance(scaled=self.scaled, cosine_weight=self.cosine_weight)
        normalized_centroids = F.normalize(self.centroids, dim=1, p=2)
        dist_fea2cen = getattr(DIST, self.distance)(embed_fea, normalized_centroids)  # [n,c+1]
        dist_cen2cen = DIST.l2(normalized_centroids, normalized_centroids)  # [c+1,c+1]

        if self.thresholds is not None:
            print(1111)
            dist_gen2cen_temp = getattr(DIST, self.distance)(generate_fea, normalized_centroids)  # [n,c+1]
            mask = dist_gen2cen_temp - self.thresholds.unsqueeze(dim=0)
            value_min, indx_min = mask.min(dim=1, keepdim=False)
            dist_gen2cen = dist_gen2cen_temp[value_min > 0, :]

        return {
            "backbone_fea": x,
            # "logits": logits,
            "embed_fea": embed_fea,
            "dist_fea2cen": dist_fea2cen,
            "dist_cen2cen": dist_cen2cen,
            "dist_gen2cen": dist_gen2cen
        }


def demo():
    x = torch.rand([10, 3, 32, 32])
    net = DFPNet('ResNet18', num_classes=10, embed_dim=64, thresholds=torch.rand(11))
    output = net(x)
    # print(output["logits"].shape)
    print(output["embed_fea"].shape)
    print(output["dist_fea2cen"].shape)
    print(output["dist_cen2cen"].shape)

    # x = torch.rand([1, 1, 28, 28])
    # net = DFPNet('LeNetPlus', num_classes=10, embed_dim=64)
    # output = net(x)
    # print(output["logits"].shape)
    # print(output["embed_fea"].shape)
    # print(output["dist_fea2cen"].shape)

# demo()
