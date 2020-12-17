import torch


def energy_hist(Out_list: torch.Tensor, Target_list:torch.Tensor, args, name:str):
    unknown_label = Target_list.max()
    unknown_list = Out_list[Target_list == unknown_label]
    known_list = Out_list[Target_list != unknown_label]
    unknown_hist = torch.histc(unknown_list, bins=args.hist_bins, min=Out_list.min().data,
                               max=Out_list.max().data)
    known_hist = torch.histc(known_list, bins=args.hist_bins, min=Out_list.min().data,
                           max=Out_list.max().data)
    if args.hist_norm:
        unknown_hist = unknown_hist/(unknown_hist.sum())
        known_hist = known_hist/(known_hist.sum())
        name += "_normed"
    print(f"{name} processed.")
