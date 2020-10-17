import torch
import torch.nn as nn
import torch.nn.functional as F


class DFPLoss(nn.Module):
    def __init__(self, alpha=1, beta=1):
        super(DFPLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss()

    def forward(self, net_out, labels):
        logits = net_out["logits"]
        loss_classify = self.ce(logits, labels)

        dist_fea2cen = net_out["dist_fea2cen"]

        batch_size, num_classes = dist_fea2cen.shape
        classes = torch.arange(num_classes, device=labels.device).long()
        labels = labels.unsqueeze(1).expand(batch_size, num_classes)
        mask = labels.eq(classes.expand(batch_size, num_classes))
        dist_within = (dist_fea2cen * mask.float()).sum(dim=1, keepdim=True)

        dist_between = F.relu(dist_within - dist_fea2cen, inplace=True)  # ensure within_distance greater others
        dist_between = dist_between.sum(dim=1, keepdim=False)
        dist_between = dist_between / (num_classes - 1.0)

        loss_within = self.alpha * (dist_within.sum()) / batch_size
        loss_between = self.beta * (dist_between.sum()) / batch_size

        loss = loss_classify + loss_within + loss_between

        return {
            "total": loss,
            "classify": loss_classify,
            "within": loss_within,
            "between": loss_between,
        }


def demo():
    n = 3
    c = 5
    dist_fea2cen = torch.rand([n, c])
    logits = torch.rand([n, c])
    label = torch.empty(3, dtype=torch.long).random_(5)
    print(label)
    loss = DFPLoss(1., 1.)
    netout = {
        "dist_fea2cen": dist_fea2cen,
        "logits": logits
    }
    dist_loss = loss(netout, label)
    print(dist_loss['total'])
    print(dist_loss['classify'])
    print(dist_loss['within'])
    print(dist_loss['between'])



# demo()
