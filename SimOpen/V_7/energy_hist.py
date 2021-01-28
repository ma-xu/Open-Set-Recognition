import torch
import os.path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def energy_hist(Out_list: torch.Tensor, Target_list:torch.Tensor, args, name:str):
    unknown_label = Target_list.max()
    unknown_list = Out_list[Target_list == unknown_label]
    known_list = Out_list[Target_list != unknown_label]
    print(f"Out_list: {Out_list}")
    print(f"Out_list.min(): {Out_list.min()}")
    print(f"Out_list.max(): {Out_list.max()}")
    unknown_hist = torch.histc(unknown_list, bins=args.hist_bins, min=Out_list.min(),
                               max=Out_list.max())
    known_hist = torch.histc(known_list, bins=args.hist_bins, min=Out_list.min(),
                           max=Out_list.max())
    if args.hist_norm:
        unknown_hist = unknown_hist/(unknown_hist.sum())
        known_hist = known_hist/(known_hist.sum())
        name += "_normed"
    if args.hist_save:
        plot_bar(unknown_hist, known_hist, args, name)
    # torch.save(
    #     {"unknown": unknown_hist,
    #      "known": known_hist,
    #      },
    #     os.path.join(args.histfolder, name + '.pkl')
    # )
    print(f"{name} processed.")


def energy_hist_sperate(known: torch.Tensor, unknown:torch.Tensor, args, name:str):
    min_value = min(known.min().data, unknown.min().data)
    max_value = max(known.max().data, unknown.max().data)
    unknown_hist = torch.histc(unknown, bins=args.hist_bins, min=min_value, max=max_value)
    known_hist = torch.histc(known, bins=args.hist_bins, min=min_value, max=max_value)
    if args.hist_norm:
        unknown_hist = unknown_hist/(unknown_hist.sum())
        known_hist = known_hist/(known_hist.sum())
        name += "_normed"
    if args.hist_save:
        plot_bar(unknown_hist, known_hist, args, name)
    # torch.save(
    #     {"unknown": unknown_hist,
    #      "known": known_hist,
    #      },
    #     os.path.join(args.histfolder, name + '.pkl')
    # )
    print(f"{name} processed.")


def plot_bar(unknown_hist, known_hist, args, name):
    x = np.arange(args.hist_bins)
    unknown_hist = unknown_hist.data.cpu().numpy()
    known_hist = known_hist.data.cpu().numpy()

    total_width, n = 0.8, 2
    width = total_width / n
    x = x - (total_width - width) / 2

    plt.bar(x, unknown_hist, width=width, label='unknown')
    plt.bar(x + width, known_hist, width=width, label='known')
    plt.legend()
    save_name = os.path.join(args.histfolder, name + '.png')
    plt.savefig(save_name, bbox_inches='tight', dpi=args.plot_quality)
    plt.close()
