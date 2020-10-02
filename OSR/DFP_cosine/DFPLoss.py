import torch
import torch.nn as nn
import torch.nn.functional as F


class DFPLoss(nn.Module):
    """Discriminant Feature Representation loss.

        # better to find a replacement that can combine these two terms.
        Loss = Sigmoid[Dis(x, gt)]+ beta*Sigmoid[-1/(class_num-1)*Sum_i(Dis(x,cls_i))]
        Args:
            num_classes (int): number of classes.
            feat_dim (int): feature dimension.
        """

    def __init__(self, beta=1, sigma=1):
        super(DFPLoss, self).__init__()
        self.beta = beta
        self.sigma = sigma

    def forward(self, dist_fea2cen, dist_cen2cen, labels):
        batch_size, num_classes = dist_fea2cen.shape
        classes = torch.arange(num_classes, device=labels.device).long()
        labels = labels.unsqueeze(1).expand(batch_size, num_classes)
        mask = labels.eq(classes.expand(batch_size, num_classes))
        dist_within = (dist_fea2cen * mask.float()).sum(dim=1, keepdim=True)

        # dist_between = (dist * (1 - mask.float()))
        dist_between = F.relu(dist_within - dist_fea2cen, inplace=True)  # ensure within_distance greater others
        dist_between = dist_between.sum(dim=1, keepdim=False)
        # dist_between = dist_between / (num_classes - 1.0)

        loss_within = (dist_within.sum()) / batch_size
        loss_between = self.beta * (dist_between.sum()) / batch_size

        loss_cen2cen = (dist_cen2cen.sum()) / (dist_cen2cen.shape[0] - 1)
        loss_cen2cen = loss_cen2cen / (dist_cen2cen.shape[0] - 1)
        loss_cen2cen = self.sigma*(1.0 - 0.5*loss_cen2cen)

        loss = loss_within + loss_between + loss_cen2cen

        return {
            "total": loss,
            "within": loss_within,
            "between": loss_between,
            "cen2cen": loss_cen2cen
        }


def demo():
    n = 3
    c =5
    dist_fea2cen = torch.rand([n, c])
    dis_cen2cen = torch.rand([c, c])
    label = torch.Tensor([1, 3, 2])
    loss = DFPLoss(1,1)
    dist_loss = loss(dist_fea2cen, dis_cen2cen, label)
    print(dist_loss['total'])
    print(dist_loss['within'])
    print(dist_loss['between'])
    print(dist_loss['cen2cen'])

demo()
