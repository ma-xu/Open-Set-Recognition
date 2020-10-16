import torch

def generater(inputs, targets,num_class,repeats=3):
    b,c,w,h = inputs.shape
    g_data = inputs.repeat(repeats, 1,1,1)  # repeat examples 3 times [n*sampler, c]
    g_data = g_data[torch.randperm(g_data.size()[0])]
    g_data = g_data.view(b,repeats,c,w,h)
    g_data = g_data.mean(dim=1, keepdim=False)
    g_label = num_class*torch.ones(b,dtype=targets.dtype)
    inputs= torch.cat([inputs,g_data],dim=0)
    targets = torch.cat([targets, g_label], dim=0)

    r = torch.randperm(inputs.size()[0])
    inputs = inputs[r]
    targets = targets[r]
    return inputs, targets







def demo():
    n = 2
    c = 5
    inputs = torch.rand([n, 1,3,3])
    targets = torch.empty(n, dtype=torch.long).random_(c)
    generater(inputs, targets,n)

# demo()
