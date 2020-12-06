import torch
import torch.nn as nn
import torch.nn.functional as F


class DFPLoss(nn.Module):
    def __init__(self, alpha=1.0, temperature=1.0, scaling=1.0):
        super(DFPLoss, self).__init__()
        self.alpha = alpha
        self.scaling = scaling
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, net_out, targets):
        sim_classification = net_out["dotproduct_fea2cen"]  # [n, class_num]; range [-1,1] greater indicates more similar.
        sim_classification = sim_classification / self.temperature
        loss_classification = self.ce(sim_classification, targets)

        dist_fea2cen = net_out["cosine_fea2cen"]
        dist_fea2cen = torch.exp(dist_fea2cen)-1.0
        # dist_fea2cen = 0.5 * (dist_fea2cen ** 2)  # 0.5*||d||^2
        batch_size, num_classes = dist_fea2cen.shape
        classes = torch.arange(num_classes, device=targets.device).long()
        labels = targets.unsqueeze(1).expand(batch_size, num_classes)
        mask = labels.eq(classes.expand(batch_size, num_classes))
        dist_within = (dist_fea2cen * mask.float()).sum(dim=1, keepdim=True)
        loss_distance = self.alpha * (dist_within.sum()) / batch_size

        loss = loss_classification + loss_distance

        return {
            "total": loss,
            "classification": loss_classification,
            "distance": loss_distance
        }


def demo():
    n = 3
    c = 5
    sim_fea2cen = torch.rand([n, c])

    label = torch.empty(3, dtype=torch.long).random_(c)
    print(label)
    loss = DFPLoss(1.)
    netout = {
        "dotproduct_fea2cen": sim_fea2cen,
        "cosine_fea2cen": sim_fea2cen
    }
    dist_loss = loss(netout, label)
    print(dist_loss)


# demo()
