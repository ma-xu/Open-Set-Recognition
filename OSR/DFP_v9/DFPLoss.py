import torch
import torch.nn as nn
import torch.nn.functional as F


class DFPLoss(nn.Module):
    def __init__(self, alpha=1):
        super(DFPLoss, self).__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, net_out, targets):
        sim_fea2cen = net_out["sim_fea2cen"]
        loss_similarity = self.ce(sim_fea2cen, targets)

        dist_fea2cen = net_out["dis_fea2cen"]
        batch_size, num_classes = dist_fea2cen.shape
        classes = torch.arange(num_classes, device=targets.device).long()
        labels = targets.unsqueeze(1).expand(batch_size, num_classes)
        mask = labels.eq(classes.expand(batch_size, num_classes))
        dist_within = (dist_fea2cen * mask.float()).sum(dim=1, keepdim=True)
        loss_distance = self.alpha * (dist_within.sum()) / batch_size

        loss = loss_similarity + loss_distance

        return {
            "total": loss,
            "similarity": loss_similarity,
            "distance": loss_distance
        }


def demo():
    n = 3
    c = 5
    dist_fea2cen = torch.rand([n, c + 1])
    dist_gen2cen = torch.rand([n, c + 1])

    label = torch.empty(3, dtype=torch.long).random_(5)
    print(label)
    loss = DFPLoss(1.)
    netout = {
        "sim_fea2cen": dist_gen2cen,
        "dis_fea2cen": dist_fea2cen
    }
    dist_loss = loss(netout, label)
    print(dist_loss['total'])
    print(dist_loss['similarity'])
    print(dist_loss['distance'])



demo()
