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
    def __init__(self, num_classes=10, feat_dim=512, beta=1, distance='l2', scaled=True):
        super(DFPLoss, self).__init__()
        self.num_classes = num_classes
        self.centers = nn.Parameter(torch.randn(self.num_classes, feat_dim))
        self.register_buffer("classes", torch.arange(self.num_classes).long())
        self.distance = distance
        assert self.distance in ['l1','l2','dotproduct']
        self.scaled = scaled
        self.beta = beta

    def forward(self, x, labels):
        batch_size = x.size(0)
        distanceFun = Distance(x, self.centers)
        dist = getattr(distanceFun, self.distance)(scaled=self.scaled) # [n, class_num]

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(self.classes.expand(batch_size, self.num_classes))
        dist_within = (dist * mask.float()).sum(dim=1,keepdim=True)

        # dist_between = (dist * (1 - mask.float()))
        dist_between = F.relu(dist_within-dist,inplace=True)  #ensure within_distance greater others
        dist_between = dist_between.sum(dim=1, keepdim=False)
        dist_between = dist_between / (self.num_classes - 1.0)



        loss_within = (dist_within.sum()) / batch_size
        loss_between = self.beta * (dist_between.sum()) / batch_size



        loss = loss_within+loss_between
        return loss, loss_within, loss_between



def demo():
    x = torch.rand([3,10])
    label = torch.Tensor([0,0,2])
    loss = DFPLoss(5,10)
    y1,y2,y3 = loss(x,label)
    print(y1)
    print(y2)
    print(y3)


demo()
