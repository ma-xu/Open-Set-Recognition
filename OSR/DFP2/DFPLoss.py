import torch
import torch.nn as nn
from Distance import Distance
import torch.nn.functional as F


class DFPLoss(nn.Module):
    """Discriminant Feature Representation loss.

        # better to find a replacement that can combine these two terms.
        Loss = Sigmoid[Dis(x, gt)]+ beta*Sigmoid[-1/(class_num-1)*Sum_i(Dis(x,cls_i))]
        Args:
            num_classes (int): number of classes.
            feat_dim (int): feature dimension.
        """

    def __init__(self, beta=1):
        super(DFPLoss, self).__init__()
        self.beta = beta

    def forward(self, dist, labels):
        batch_size, num_classes = dist.shape
        classes = torch.arange(num_classes, device=labels.device).long()
        labels = labels.unsqueeze(1).expand(batch_size, num_classes)
        mask = labels.eq(classes.expand(batch_size, num_classes))
        dist_within = (dist * mask.float()).sum(dim=1, keepdim=True)

        # dist_between = (dist * (1 - mask.float()))
        dist_between = F.relu(dist_within - dist, inplace=True)  # ensure within_distance greater others
        dist_between = dist_between.sum(dim=1, keepdim=False)
        dist_between = dist_between / (num_classes - 1.0)

        loss_within = (dist_within.sum()) / batch_size
        loss_between = self.beta * (dist_between.sum()) / batch_size

        loss = loss_within + loss_between

        return {
            "total": loss,
            "within": loss_within,
            "between": loss_between,
        }


def demo():
    x = torch.rand([3, 5])
    label = torch.Tensor([1, 3, 2])
    loss = DFPLoss(5)
    dist_loss = loss(x, label)
    print(dist_loss['total'])
    print(dist_loss['within'])
    print(dist_loss['between'])


# demo()
