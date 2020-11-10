import torch
from Utils import progress_bar

def get_stat(net, testloader, device, args):

    Features = {i: [] for i in range(args.train_class_num)}
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            out = net(inputs)
            out = (out['embed_fea']).unsqueeze(dim=1)
            for i in targets:
                target = (targets[i]).item()
                (Features[target]).append(out[i])
            progress_bar(batch_idx, len(testloader), "calculating statistics ...")

        list_std = []
        list_mean = []
        for i in range(args.train_class_num):
            feature = torch.cat(Features[i], dim=0)
            list_std.append(torch.std(feature,dim=0,keepdim=True))
            list_mean.append(torch.mean(feature, dim=0, keepdim=True))
    list_std = torch.cat(list_std,dim=0)
    list_mean = torch.cat(list_mean, dim=0)
    return {"std":list_std,"mean":list_mean}







