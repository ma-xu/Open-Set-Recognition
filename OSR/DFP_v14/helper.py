import torch
from Utils import progress_bar

def get_stat(net, testloader, device, args):
    correct = 0
    total = 0
    Features = {i: [] for i in range(args.train_class_num)}
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            out = net(inputs)
            out = out['embed_fea']
            for target in targets:
                print(f"{type(target)}, {target}, {type(out)}, , {out.shape}")
                # label = targets==i
                # label =
                # dist = dist_fea2cen[i, label]
                # results[label.item()]["distances"].append(dist)


            progress_bar(batch_idx, len(testloader), "calculating statistics ...")


