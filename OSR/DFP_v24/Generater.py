import torch
import numpy as np

import random
import torch.nn.functional as F
from random import shuffle
from math import ceil


class CGDestimator():
    def __init__(self, stat=None):
        super(CGDestimator, self).__init__()
        if stat is not None:
            self.mean = stat["mean"]  # [class_number, channel]
            self.std = stat["std"]  # [class_number, channel]
            self.fea_bank = stat["fea_bank"]  # [class_number, 1024, channel]
            self.gen_fea = self.generator()
            self.gen_n = self.gen_fea.shape[0]

    def generator(self):
        fea_norm_disted = (self.fea_bank - self.mean.unsqueeze(dim=1)) / (self.std.unsqueeze(dim=1))
        cls_num, bank_size, channel = self.fea_bank.shape
        fea_norm_disted = fea_norm_disted.unsqueeze(dim=0).expand([cls_num, cls_num, bank_size, channel])
        fea_new = fea_norm_disted * (self.std.unsqueeze(dim=1).unsqueeze(dim=1)) \
                  + self.mean.unsqueeze(dim=1).unsqueeze(dim=1)

        mask = torch.arange(cls_num)
        mask = mask * cls_num + mask
        mask = mask.tolist()
        mask2 = torch.arange(cls_num * cls_num)
        mask2 = mask2.tolist()
        [mask2.remove(x) for x in mask]

        fea_new = fea_new.view([cls_num*cls_num, bank_size, channel])
        fea_new = fea_new[mask2]

        fea_new = fea_new.view(-1,channel)
        return fea_new

    def sampler(self,gap):
        inds = torch.randint(0, self.gen_n, [2*gap.shape[0]])
        out = self.gen_fea[inds]
        return out


    def generator2(self, gap):
        batch_size = gap.shape[0]
        class_num, channel_num = self.mean.size()

        n = max(ceil(batch_size / class_num), 3)

        noise = torch.randn(n, class_num * channel_num).to(gap.device)
        channel_mean = self.mean.view(-1)
        channel_std = self.std.view(-1)
        noise = (noise - noise.mean(dim=0, keepdim=True)) / (noise.std(dim=0, keepdim=True))
        noise = noise * channel_std + channel_mean

        noise = noise.reshape([n, class_num, channel_num])
        noise = noise.reshape([n * class_num, channel_num])
        noise = noise[torch.randperm(noise.size()[0])]
        noise = noise[0:batch_size]
        noise = noise.clone().detach()
        # data = gap + data - gap

        return noise


def demo():



    class_number = 5
    channel = 16
    stat = {
        "mean": torch.rand([class_number, channel]),
        "std": torch.rand([class_number, channel]),
        "fea_bank": torch.rand([class_number, 1024, channel]),
    }
    gap = torch.rand(4, channel)

    estimator = CGDestimator(stat)
    estimator.sampler(gap)


demo()
