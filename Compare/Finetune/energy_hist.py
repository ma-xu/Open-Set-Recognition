import torch
import os.path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_listhist(hists: list, args, labels: list, name: str):
    # plot_listhist([a, b, c, d], args, ["a", "b", "c", "d"], "test")
    assert len(hists) > 1
    assert len(hists) == len(labels)
    max_value = [hist.max() for hist in hists]
    max_value = max(max_value)
    min_value = [hist.min() for hist in hists]
    min_value = min(min_value)

    total_width, n = 0.8, hists.__len__()
    width = total_width / n
    x = np.arange(args.hist_bins)
    x = x - (total_width - width) / (n-1)

    for i in range(0, len(hists)):
        hist = torch.histc(hists[i], bins=args.hist_bins, min=min_value, max=max_value)
        hist = hist.data.cpu().numpy()
        if args.hist_norm:
            hist = hist / (hist.sum())
        plt.bar(x + width*i, hist, width=width, label=labels[i])
    plt.legend()
    save_name = os.path.join(args.checkpoint, name + '.pdf')
    plt.savefig(save_name,bbox_inches='tight', dpi=200)
    plt.close()

