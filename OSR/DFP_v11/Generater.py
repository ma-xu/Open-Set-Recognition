import torch
import numpy as np
import torch.nn.functional as F
from random import shuffle


def generater_input(inputs, targets, args, repeats=4, reduce=16):
    b, c, w, h = inputs.shape
    if b != args.bs:
        return inputs, targets
    g_data = inputs.repeat(repeats, 1, 1, 1)  # repeat examples 3 times [n*sampler, c]
    g_data = g_data[torch.randperm(g_data.size()[0])]
    g_data = g_data.view(-1, b // reduce, c, w, h)
    g_data = g_data.mean(dim=0, keepdim=False)
    g_label = args.train_class_num * torch.ones(b // reduce, dtype=targets.dtype)
    inputs = torch.cat([inputs, g_data], dim=0)
    targets = torch.cat([targets, g_label], dim=0)

    r = torch.randperm(inputs.size()[0])
    inputs = inputs[r]
    targets = targets[r]
    return inputs, targets

def generater_unknown(inputs, targets, args, repeats=4, reduce=16):
    b, c, w, h = inputs.shape

    number = b // reduce if b == args.bs else b

    g_data = inputs.repeat(repeats, 1, 1, 1)  # repeat examples 3 times [n*sampler, c]
    g_data = g_data[torch.randperm(g_data.size()[0])]
    g_data = g_data.view(-1, number, c, w, h)
    g_data = g_data.mean(dim=0, keepdim=False)
    return g_data



def generater_gap(gap,batchsize=32):
    # generated a random gap doesn't require gradient
    b, c = gap.size()
    mem = gap.clone().detach()
    mem = mem.view(-1)
    mem = mem[torch.randperm(mem.size()[0])]
    mem = mem.view([b, c])
    if batchsize < b:
        mem = mem[:batchsize]
    mem = mem.to(gap.device)
    return mem


def generater_gap2(gap):
    # generated a random gap doesn't require gradient
    b, c = gap.size()
    mem = gap.clone().detach()
    mem = mem.view(-1)
    mem = mem[torch.randperm(mem.size()[0])]
    mem = mem.view([b, c])

    mem = mem.to(gap.device)
    # directly passing the gradient.
    generate = gap + mem - gap
    return generate

def generater_gap3(gap,shuffle_rate=2):
    # generated a random gap doesn't require gradient
    b, c = gap.size()
    mem = gap.clone().detach()
    mem = mem.permute(1,0)  #c,b
    mem = mem.tolist()
    selected_num = max (4, c//shuffle_rate)
    selected_channels = torch.empty(selected_num, dtype=torch.long).random_(c)
    for i in selected_channels:
        m = mem[i]
        shuffle(m)
        mem[i] = m

    mem = torch.Tensor(mem)
    mem = mem.permute(1,0)
    mem = mem.to(gap.device)
    # directly passing the gradient.
    generate = gap + mem - gap
    return generate


def demo():
    n = 16
    c = 5
    inputs = torch.rand([n, 1, 3, 3])
    targets = torch.empty(n, dtype=torch.long).random_(c)
    generater_input(inputs, targets, n)

# demo()

def demo_gap():
    gap = torch.rand([3,6],requires_grad=True)
    generater_gap(gap)

# demo_gap()

def demo_shuffle():
    gap = torch.arange(1,11,dtype=float).unsqueeze(dim=-1).expand(10,64)
    gap.requires_grad = True

    generate = generater_gap3(gap)

    print(generate)
    print(generate.requires_grad)

# demo_shuffle()
