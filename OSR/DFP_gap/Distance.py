import torch
import math
import torch.nn.functional as F


class Distance(torch.nn.Module):
    # Distance should return a tensor with shape [n, num_classes]
    def __init__(self, scaled=True, cosine_weight=1):
        super(Distance, self).__init__()
        self.scaled = scaled
        self.cosine_weight = cosine_weight

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
        centroids = centroids.unsqueeze(0).expand(sample_num, class_num, dim_num)
        features = features.unsqueeze(1).expand(sample_num, class_num, dim_num)
        distance = F.cosine_similarity(features, centroids, dim=2)

        # re-range the range [-1,1] tp [0,1]
        # after re-ranging, distance 0 means same, 0.5 means orthogonality/decorrelation, 1 means opposite.
        # referring https://en.wikipedia.org/wiki/Cosine_similarity
        distance = (1.0 - distance) / 2.0
        distance = self.cosine_weight * distance

        # We don't consider scled in consine-similarity.
        # if self.scaled:
        #     distance = distance / math.sqrt(dim_num)
        return distance


class Similarity(torch.nn.Module):

    def __init__(self, scaled=True):
        super(Similarity, self).__init__()
        self.scaled = scaled

    def l2(self, features, centroids):
        # We can directly call torch.dist(), or cdist(), or pairwise_distance
        _, dim_num = features.shape
        centroids = centroids.unsqueeze(0)
        features = features.unsqueeze(0)
        distance = torch.cdist(features, centroids, 2).squeeze(0)
        if self.scaled:
            scale = math.sqrt(features.size(-1))
            distance = distance / scale
        sim = 1/(1.0+distance)
        return sim

    def cosine(self, features, centroids):
        sample_num, dim_num = features.shape
        class_num = centroids.shape[0]
        centroids = centroids.unsqueeze(0).expand(sample_num, class_num, dim_num)
        features = features.unsqueeze(1).expand(sample_num, class_num, dim_num)
        sim = F.cosine_similarity(features, centroids, dim=2)
        return sim

    def dotproduct(self, features, centroids):
        centroids = centroids.permute(1,0)
        sim = torch.matmul(features, centroids)
        if self.scaled:
            scale = math.sqrt(features.size(-1))
            sim = sim / scale
        return sim


def demo():
    feature = torch.rand(3, 10)
    centroids = torch.rand(5, 10)
    distance = Distance()
    metric = 'cosine'
    print(getattr(distance, metric)(feature, centroids))
    # print(distance.l2().shape)

    similarity = Similarity()
    metric = 'dotproduct'
    print(getattr(similarity, metric)(feature, centroids))

# demo()
