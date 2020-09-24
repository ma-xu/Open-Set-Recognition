import torch
import math
import torch.nn.functional as F

class Distance(torch.nn.Module):
    # Distance should return a tensor with shape [n, num_classes]
    def __init__(self, features, centroids):
        super(Distance, self).__init__()
        self.features = features
        self.centroids = centroids
        assert len(self.features.shape) == 2  # [n, feat_dim]
        assert len(self.centroids.shape) == 2  # [num_classes, feat_dim]
        self.sample_num = self.features.size(0)
        self.dim_num = self.features.size(1)
        self.class_num = self.centroids.size(0)

    def dotproduct(self, scaled=True):
        ### DEPRECATED. dotproduct can not be used as similarity or distance. See cosine similarity.
        centroids_t = self.centroids.permute(1, 0)
        distance = torch.mm(self.features,centroids_t)
        if scaled:
            distance = distance/math.sqrt(self.dim_num)
        return distance

    def l1(self, scaled=True):
        centroids_t = self.centroids.permute(1, 0)
        centroids_t = centroids_t.unsqueeze(0).expand(self.sample_num, self.dim_num, self.class_num)
        features = self.features.unsqueeze(-1).expand(self.sample_num, self.dim_num, self.class_num)
        distance = abs(features-centroids_t)
        distance = distance.sum(dim=1,keepdim=False)
        if scaled:
            distance = distance/math.sqrt(self.dim_num)
        return distance


    def l2(self, scaled=True):
        # We can directly call torch.dist(), or cdist(), or pairwise_distance
        centroids = self.centroids.unsqueeze(0)
        features = self.features.unsqueeze(0)
        distance = torch.cdist(features, centroids, 2).squeeze(0)
        if scaled:
            distance = distance/math.sqrt(self.dim_num)
        return distance

    def cosine(self, scaled=False):
        centroids = self.centroids.unsqueeze(0).expand(self.sample_num,self.class_num, self.dim_num)
        features = self.features.unsqueeze(1).expand(self.sample_num, self.class_num, self.dim_num)
        distance = F.cosine_similarity(features,centroids, dim=2)
        if scaled:
            distance = distance / math.sqrt(self.dim_num)
        return distance


def demo():
    feature = torch.rand(10,20)
    centroids = torch.rand(5,20)
    distance = Distance(feature,centroids)
    metric = 'cosine'
    scaled = True
    print(getattr(distance, metric)(scaled=scaled))
    print(distance.l2().shape)

# demo()
