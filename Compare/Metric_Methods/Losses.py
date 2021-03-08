import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self,centerloss_weight = 0.003, num_classes=10):
        super(CenterLoss, self).__init__()
        self.centerloss_weight = centerloss_weight


    def forward(self, inputs, labels):
        centroids = inputs["centroids"]
        x = inputs["embed_fea"]
        batch_size = x.size(0)
        num_classes = centroids.size(0)
        classes = torch.arange(num_classes).long()
        classes.to(labels.device)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, num_classes) + \
                  torch.pow(centroids, 2).sum(dim=1, keepdim=True).expand(num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, centroids.t())

        labels = labels.unsqueeze(1).expand(batch_size, num_classes)
        mask = labels.eq(self.classes.expand(batch_size, num_classes))

        dist = distmat * mask.float()
        center_loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        center_loss = self.centerloss_weight * center_loss
        softmax_loss = F.cross_entropy(inputs["dotproduct_fea2cen"], labels)

        return {
            "loss":softmax_loss + center_loss,
            "softmax_loss": softmax_loss,
            "center_loss": center_loss
        }


class SoftmaxLoss(nn.Module):
    def __init__(self):
        super(SoftmaxLoss, self).__init__()

    def forward(self, inputs, labels):
        loss = F.cross_entropy(inputs["dotproduct_fea2cen"], labels)
        return {
            "loss": loss
        }


class ArcFaceLoss(nn.Module):
    def __init__(self, scaling=32, m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.scaling = scaling
        self.m = m
        self.ce = nn.CrossEntropyLoss()
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # ensure cos(theta+m) decreases in the range of (0,pi)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, net_out, targets):
        cosine = net_out["cosine_fea2cen"]
        cosine = cosine.clamp(-1, 1)
        sine = torch.sqrt(torch.max(1.0 - torch.pow(cosine, 2), torch.ones_like(cosine) * 1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, targets.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scaling
        loss = self.ce(output, targets)

        return {
            "loss": loss,
            "output": output
        }


class NormFaceLoss(nn.Module):
    def __init__(self,scaling=16):
        super(NormFaceLoss, self).__init__()
        self.scaling = scaling
        self.ce = nn.CrossEntropyLoss()

    def forward(self, net_out, targets):
        output = self.scaling * net_out["cosine_fea2cen"]  # [n, class_num]; range [-1,1] greater indicates more similar.
        loss = self.ce(output, targets)
        return {
            "loss": loss,
            "output":output
        }


class PSoftmaxLoss(nn.Module):
    def __init__(self):
        super(PSoftmaxLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, net_out, targets):
        normweight_fea2cen = net_out["normweight_fea2cen"]  # [n, class_num]; range [-1,1] greater indicates more similar.
        loss = self.ce(normweight_fea2cen, targets)
        return {
            "loss": loss
        }
