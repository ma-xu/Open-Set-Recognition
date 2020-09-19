import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter

import pdb

__all__=['CosNorm_Classifier', 'DotProduct_Classifier', 'MetaEmbedding_Classifier']
"""
CosNorm Classifier
"""
class CosNorm_Classifier(nn.Module):
    def __init__(self, in_dims, out_dims, scale=16, margin=0.5, init_std=0.001):
        super(CosNorm_Classifier, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.scale = scale
        self.margin = margin
        self.weight = Parameter(torch.Tensor(out_dims, in_dims))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, *args):
        norm_x = torch.norm(input.clone(), 2, 1, keepdim=True)
        ex = (norm_x / (1 + norm_x)) * (input / norm_x)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        return torch.mm(self.scale * ex, ew.t()),None


# def create_model(in_dims=512, out_dims=1000):
#     print('Loading Cosine Norm Classifier.')
#     return CosNorm_Classifier(in_dims=in_dims, out_dims=out_dims)



"""
Dotproduct Classifier
"""
# from utils import *


class DotProduct_Classifier(nn.Module):

    def __init__(self, feat_dim=512, num_classes=1000, *args):
        super(DotProduct_Classifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x, *args):
        y = self.fc(x)
        return y, x


# def create_model(feat_dim, num_classes=1000, stage1_weights=False, dataset=None, test=False, *args):
#     print('Loading Dot Product Classifier.')
#     clf = DotProduct_Classifier(num_classes, feat_dim)
#
#     if not test:
#         if stage1_weights:
#             assert (dataset)
#             print('Loading %s Stage 1 Classifier Weights.' % dataset)
#             # clf.fc = init_weights(model=clf.fc,
#             #                       weights_path='./logs/%s/stage1/final_model_checkpoint.pth' % dataset,
#             #                       classifier=True)
#         else:
#             print('Random initialized classifier weights.')
#
#     return clf

"""
Meta-embedding Classifier
"""


class MetaEmbedding_Classifier(nn.Module):

    def __init__(self, feat_dim=2048, num_classes=1000):
        super(MetaEmbedding_Classifier, self).__init__()
        self.num_classes = num_classes
        self.fc_hallucinator = nn.Linear(feat_dim, num_classes)
        self.fc_selector = nn.Linear(feat_dim, feat_dim)
        self.cosnorm_classifier = CosNorm_Classifier(feat_dim, num_classes)

    def forward(self, x, centroids, *args):
        # storing direct feature
        direct_feature = x.clone()

        batch_size = x.size(0)
        feat_size = x.size(1)

        # set up visual memory
        x_expand = x.clone().unsqueeze(1).expand(-1, self.num_classes, -1)
        centroids_expand = centroids.clone().unsqueeze(0).expand(batch_size, -1, -1)
        keys_memory = centroids.clone()

        # computing reachability
        dist_cur = torch.norm(x_expand - centroids_expand, 2, 2)
        values_nn, labels_nn = torch.sort(dist_cur, 1)
        scale = 10.0
        reachability = (scale / values_nn[:, 0]).unsqueeze(1).expand(-1, feat_size)

        # computing memory feature by querying and associating visual memory
        values_memory = self.fc_hallucinator(x.clone())
        values_memory = values_memory.softmax(dim=1)
        memory_feature = torch.matmul(values_memory, keys_memory)

        # computing concept selector
        concept_selector = self.fc_selector(x.clone())
        concept_selector = concept_selector.tanh()
        x = reachability * (direct_feature + concept_selector * memory_feature)

        # storing infused feature
        infused_feature = concept_selector * memory_feature

        logits = self.cosnorm_classifier(x)

        return logits, [direct_feature, infused_feature]


# def create_model(feat_dim=2048, num_classes=1000, stage1_weights=False, dataset=None, test=False, *args):
#     print('Loading Meta Embedding Classifier.')
#     clf = MetaEmbedding_Classifier(feat_dim, num_classes)
#
#     if not test:
#         if stage1_weights:
#             assert (dataset)
#             print('Loading %s Stage 1 Classifier Weights.' % dataset)
#             # clf.fc_hallucinator = init_weights(model=clf.fc_hallucinator,
#             #                                    weights_path='./logs/%s/stage1/final_model_checkpoint.pth' % dataset,
#             #                                    classifier=True)
#         else:
#             print('Random initialized classifier weights.')
#
#     return clf
