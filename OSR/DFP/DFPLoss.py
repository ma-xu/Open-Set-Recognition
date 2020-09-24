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
        dist_within = (dist * mask.float()).sum(dim=1, keepdim=False)
        loss_within = (torch.sigmoid(dist_within).sum()) / batch_size

        """ Version 1: L2-distance to other labels 
        Function: beta*Sigmoid[-1/(class_num-1)*Sum_i(Dis(x,cls_i))]
        # A question: distance to all other centroids or the closed non-gt centroid.
        dist_between = (-dist*(1-mask.float())).sum(dim=1, keepdim=False)  # convert max to min
        dist_between = dist_between/(self.num_classes-1.0)
        loss_between = self.beta * (torch.sigmoid(dist_between).sum()) / batch_size
        """

        """Version 2: Cosine similarity to other labels()
        Function: beta*1/(class_num-1)*Sum_i(Dis(x,cls_i))
        similarity = getattr(distanceFun, self.similarity)(scaled=True) # [n, class_num]
        sim_between = (similarity*(1-mask.float())).sum(dim=1, keepdim=False)
        sim_between = sim_between/(self.num_classes-1.0)
        loss_between = self.beta *(sim_between.sum()) / batch_size
        """

        """Version 3: L2-distance to other labels
        """
        distanceFun2 = Distance(self.centers, self.centers)
        dist2 = getattr(distanceFun2, self.distance)(scaled=self.scaled)  # [n, class_num]
        dist_between = -(dist2).sum(dim=1, keepdim=False)  # convert max to min
        dist_between = dist_between / (self.num_classes - 1.0)
        loss_between = self.beta * (torch.sigmoid(dist_between).sum()) / batch_size


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


# demo()
