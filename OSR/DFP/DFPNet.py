import torch
import torch.nn as nn
import torch.nn.functional as F
import backbones.cifar as models
from Distance import Distance


class DFPNet(nn.Module):
    def __init__(self, backbone='ResNet18', num_classes=1000, backbone_fc=False,
                 include_dist=False, embed_dim=None, embed_reduction=8):
        """

        :param backbone: the backbone architecture, default ResNet18
        :param num_classes: known classes
        :param backbone_fc: includes FC layers in backbone, default false.
        :param include_dist: if include the distance branch, default false to get traditional backbone architecture.
        :param embed_dim: if include the embedding layer and tell the embedding dimension.
        :param embed_reduction: for embedding reduction in SENet style. May deprecated.
        """
        super(DFPNet, self).__init__()
        assert backbone_fc == False # drop out the classifier layer.
        # num_classes = num_classes would be useless if backbone_fc=False
        self.backbone = models.__dict__[backbone](num_classes=num_classes, backbone_fc=backbone_fc)
        feat_dim = self.get_backbone_last_layer_out_channel()  # get the channel number of backbone output
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.include_dist = include_dist
        if embed_dim and include_dist:
            # Embedding layer could be modified to a whitened feature map like DNL.
            self.embeddingLayer = nn.Sequential(
                # embed_reduction just for parameter reduction, not attention mechanism.
                nn.Linear(feat_dim,embed_dim//embed_reduction),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim//embed_reduction, embed_dim)
            )

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

        gap = F.adaptive_avg_pool2d(x,1)
        gap = F.relu(gap.view(gap.size(0), -1), inplace=True)

        # processing the clssifier branch
        logits = self.classifier(gap)
        # processing the distance branch
        embed_fea = None
        if self.include_dist :
            embed_fea = self.embeddingLayer(gap) if hasattr(self, 'embeddingLayer') else gap
            """Calculate distance in DFPLoss"""
            # Dist = Distance(embed_fea, centroids)
            # dis = getattr(Dist, self.distance)(scaled=self.scaled) # return [n, num_classes]
        return logits, embed_fea


def demo():
    x = torch.rand([1, 3, 32, 32])
    net = DFPNet('ResNet18',num_classes=100, embed_dim=64, include_dist=True)
    logits, embed_fea= net(x)
    print(logits.shape)
    print(embed_fea)


# demo()
