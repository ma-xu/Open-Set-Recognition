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
            out = (out['embed_fea']).unsqueeze(dim=1)
            for i in targets:
                target = (targets[i]).item()
                (Features[target]).append(out[i])

        for i in range(args.train_class_num):
            feature = torch.cat(Features[i], dim=0)
            print("____"*5)
            print(torch.std(feature,dim=1))
            print(torch.mean(feature, dim=1))


                # label = targets==i
                # label =
                # dist = dist_fea2cen[i, label]
                # results[label.item()]["distances"].append(dist)


            progress_bar(batch_idx, len(testloader), "calculating statistics ...")


