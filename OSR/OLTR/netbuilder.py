import sys
sys.path.append("../..")
import backbones.cifar as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier import *
from oltr import ModulatedAttLayer


class Network(nn.Module):
    def __init__(self, backbone='ResNet18', embed_dim=512, num_classes=1000,
                 use_fc=False, attmodule=True, classifier='dotproduct', backbone_fc=False, data_shape=4):
        super(Network, self).__init__()
        assert backbone_fc == False # For OLTR, we remove the backbone_FC layer.
        # here for backbone
        self.use_fc = use_fc
        self.backbone = models.__dict__[backbone](num_classes=embed_dim,backbone_fc=backbone_fc)
        feat_dim = self.get_backbone_last_layer_out_channel()
        # here for attmodule
        if attmodule:
            self.att = ModulatedAttLayer(feat_dim, height=data_shape, width=data_shape)

        # here for use fc
        if self.use_fc:
            self.fc_add = nn.Linear(feat_dim, 512)
            feat_dim = embed_dim


        # here for classifier
        classifier_map = {
            "dotproduct": DotProduct_Classifier(feat_dim, num_classes),
            "cosnorm": CosNorm_Classifier(feat_dim, num_classes),
            "metaembedding": MetaEmbedding_Classifier(feat_dim, num_classes)
        }
        self.classifier = classifier_map[classifier]

    def get_backbone_last_layer_out_channel(self):
        last_layer = list(self.backbone.children())[-1]
        while (not isinstance(last_layer, nn.Conv2d)) and \
                (not isinstance(last_layer, nn.Linear)) and \
                (not isinstance(last_layer, nn.BatchNorm2d)):

                temp_layer = list(last_layer.children())[-1]
                if isinstance(temp_layer, nn.Sequential) and len(list(temp_layer.children()))==0:
                    temp_layer = list(last_layer.children())[-2]
                last_layer = temp_layer
        if isinstance(last_layer, nn.BatchNorm2d):
            return last_layer.num_features
        else:
            return last_layer.out_channels

    def forward(self, x):
        x = self.backbone(x)
        feature_maps = None
        if hasattr(self, 'att'):
            x, feature_maps = self.att(x)
        x = F.adaptive_avg_pool2d(x,1)
        x = x.view(x.size(0), -1)
        if self.use_fc:
            x = F.relu(self.fc_add(x))

        y,fea = self.classifier(x)
        # fea is especially for meta-embedding classifier: direct_feature, infused_feature;
        # for dotproduct classifier, fea is the input (to calculate the centroids)
        # feature_maps is for attention: [x, spatial_att, mask]
        # x is the input for classifier, for centroids loss.
        return y,fea, feature_maps, x


def demo():
    # this demo didn't test metaembedding, should works if defined the centroids.
    x = torch.rand([1, 3, 32, 32])
    net = Network('ResNet18', 512, 50, use_fc=True, attmodule=True)
    y, fea, feature_maps,x = net(x)
    print(y.shape)


demo()
