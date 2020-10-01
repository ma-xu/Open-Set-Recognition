import torch
import math
import torch.nn.functional as F

class Distance(torch.nn.Module):
    # Distance should return a tensor with shape [n, num_classes]
    def __init__(self, scaled=True):
        super(Distance, self).__init__()
        self.scaled = scaled


    def l1(self, features, centroids):
        # We can directly call torch.dist(), or cdist(), or pairwise_distance
        _, dim_num = features.shape
        centroids = centroids.unsqueeze(0)
        features = features.unsqueeze(0)
        distance = torch.cdist(features, centroids, 1).squeeze(0)
        if self.scaled:
            distance = distance / math.sqrt(dim_num)
        return distance

    def l2(self, features, centroids):
        # We can directly call torch.dist(), or cdist(), or pairwise_distance
        _, dim_num = features.shape
        centroids = centroids.unsqueeze(0)
        features = features.unsqueeze(0)
        distance = torch.cdist(features, centroids, 2).squeeze(0)
        if self.scaled:
            distance = distance / math.sqrt(dim_num)
        return distance

    def cosine(self, features, centroids):
        sample_num, dim_num = features.shape
        class_num = centroids.shape[0]
        centroids = centroids.unsqueeze(0).expand(sample_num,class_num, dim_num)
        features = features.unsqueeze(1).expand(sample_num, class_num, dim_num)
        distance = F.cosine_similarity(features,centroids, dim=2)
        if self.scaled:
            distance = distance / math.sqrt(dim_num)
        return distance


def demo():
    feature = torch.rand(3,10)
    centroids = torch.rand(5,10)
    distance = Distance()
    metric = 'cosine'
    print(getattr(distance, metric)(feature,centroids))
    # print(distance.l2().shape)

# demo()

