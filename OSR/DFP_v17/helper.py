import torch
from Utils import progress_bar

def get_gap_stat(net, trainloader, device, args):

    Result = {i: {'fea_bank': []} for i in range(args.train_class_num)}
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            out = net(inputs)
            out = (out['gap']).unsqueeze(dim=1)
            for i in range(out.shape[0]):
                target = (targets[i]).item()
                (Result[target]["fea_bank"]).append(out[i])

            # start check bank storage
            # we believe 1024 samples in each class is representive enough.
            for i in range(args.train_class_num):
                if len(Result[i]["fea_bank"])>1024:
                    Result[i]["fea_bank"] = Result[i]["fea_bank"][:1024]
            # end check bank storage

            progress_bar(batch_idx, len(trainloader), "calculating statistics ...")

        list_std = []
        list_mean = []
        list_feature = []
        for i in range(args.train_class_num):
            feature = torch.cat(Result[i]["fea_bank"], dim=0)
            list_std.append(torch.std(feature,dim=0,keepdim=True))
            list_mean.append(torch.mean(feature, dim=0, keepdim=True))
            list_feature.append(feature.unsqueeze(dim=0))
    list_std = torch.cat(list_std,dim=0)  # [class_num, channel]
    list_mean = torch.cat(list_mean, dim=0)  # [class_num, channel]
    list_feature = torch.cat(list_feature, dim=0)  # [class_num, 1024, channel]
    print(f"list_std shape: {list_std.shape}")
    print(f"list_mean shape: {list_mean.shape}")
    print(f"list_feature shape: {list_feature.shape}")
    return {"std":list_std,"mean":list_mean, "fea_bank": list_feature}







