
import os
import math

import torch

a = torch.normal(1, 0.4, size=(1,80000))
b = torch.normal(-1, 0.44, size=(1,40000))
_min = min(a.min(),b.min())
_max = max(a.max(),b.max())
d1 = torch.histc(a,60, _min,_max,)
d2 = torch.histc(b,60, _min,_max,)
for i in range(0,60):
    print(f"{d1[i].item()}\t{d2[i].item()}")
#
# print("\n\n\n\n\n")
#
# for i in d2:
#     print(i.item())
