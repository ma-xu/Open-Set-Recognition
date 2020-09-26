'''LeNetPlus in PyTorch.
Specifically, designed for MNIST dataset.

Reference:
[1] Wen, Yandong, et al. "A discriminative feature learning approach for deep face recognition."
European conference on computer vision. Springer, Cham, 2016.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LeNetPlus']


class LeNetPlus(nn.Module):
    def __init__(self, num_classes=10, backbone_fc=True):
        super(LeNetPlus, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.prelu1_2 = nn.PReLU()

        self.conv2_1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.prelu2_2 = nn.PReLU()

        self.conv3_1 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.prelu3_2 = nn.PReLU()

        if backbone_fc:
            self.linear = nn.Sequential(
                nn.Linear(128 * 3 * 3, 2),
                nn.PReLU(),
                nn.Linear(2, num_classes)
            )

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)

        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)

        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 128 * 3 * 3)
        # for unified style for DFPNet
        out = x.unsqueeze(dim=-1).unsqueeze(dim=-1)

        # return the original feature map if no FC layers.
        if hasattr(self, 'linear'):
            out = F.adaptive_avg_pool2d(out, 1)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out


def demo():
    net = LeNetPlus(num_classes=10, backbone_fc=False)
    y = net(torch.randn(2, 1, 28, 28))
    print(y.size())

# demo()
