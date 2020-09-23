import torch
import math

class Distance:
    # Distance should return a tensor with shape [n, num_classes]
    def __init__(self, features,centroids):
        self.features = features
        self.centroids = centroids
        assert len(self.features.shape) == 2  # [n, feat_dim]
        assert len(self.centroids.shape) == 2  # [num_classes, feat_dim]

    def dotproduct(self, scaled=True):
        centroids_T = self.centroids.permute(1, 0)
        distance = torch.mm(self.features,centroids_T)
        if scaled:
            distance = distance/math.sqrt(self.features.size(1))
        return self.features

    def l1(self, scaled=True):
        return self.features

    def l2(self, scaled=True):
        return self.features

    def cosine(self):
        return self.features


def demo():
    feature = torch.rand(10,20)
    centroids = torch.rand(5,20)
    distance = Distance(feature,centroids)
    print(distance.dotproduct())

demo()
