import torch

def generat_rand_feature( gap, sampler=6):
    # generate a tensor with same shape as gap.
    n, c = gap.shape
    pool = gap.repeat(sampler, 1)  # repeat examples 3 times [n*sampler, c]
    pool_random = pool[torch.randperm(pool.size()[0])]
    pool_random = pool_random.view(sampler, n, c)
    pool_random = pool_random.mean(dim=0, keepdim=False)
    return pool_random

gap = torch.tensor([[1., 1., 1., 1., 1.,],
                    [2., 2., 2., 2., 2.,],
                    [3., 3., 3., 3., 3.,],
                    [4., 4., 4., 4., 4.,]
                    ])

print(gap.shape)

mm = generat_rand_feature(gap)
print(mm)
