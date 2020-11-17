import torch
import torch.nn as nn
import torch.nn.functional as F


class DSVDDLoss(nn.Module):
    def __init__(self):
        super(DSVDDLoss, self).__init__()
        self.nu = 0.1

    def forward(self, net_out):
        distance = net_out["distance"]
        radius = net_out["radius"]

        # loss for soft-boundary (we didn't employ this)
        # loss = radius ** 2 + (1 / self.nu) * (distance**2).mean()

        # loss for one-class
        loss = 0.5*(distance**2).mean()

        return loss
