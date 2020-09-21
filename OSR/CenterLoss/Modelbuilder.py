import sys
sys.path.append("../..")
import backbones.cifar as models
import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, backbone='ResNet18', num_classes=1000):
        super(Network, self).__init__()
        self.backbone = models.__dict__[backbone](num_classes=num_classes,backbone_fc=False)
        self.dim = self.get_backbone_last_layer_out_channel()
        self.classifier = nn.Linear(self.dim, num_classes)

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
        feature = self.backbone(x)
        feature = F.adaptive_avg_pool2d(feature,1)
        feature = F.relu(feature.view(x.size(0), -1),inplace=True)
        logits = self.classifier(feature)
        return feature, logits


def demo():
    # this demo didn't test metaembedding, should works if defined the centroids.
    x = torch.rand([1, 3, 32, 32])
    net = Network('ResNet18',  50)
    feature, logits = net(x)
    print(feature.shape)
    print(logits.shape)


# demo()
