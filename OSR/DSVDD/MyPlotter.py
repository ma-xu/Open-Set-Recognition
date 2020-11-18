import torch
import torch.nn as nn
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os



def plot_feature(net, args, plotloader, device, dirname, epoch=0, plot_quality=150):
    plot_features = []
    plot_labels = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(plotloader):
            inputs, targets = inputs.to(device), targets.to(device)
            out = net(inputs)
            embed_fea = out["embed_fea"]
            try:
                embed_fea = embed_fea.data.cpu().numpy()
                targets = targets.data.cpu().numpy()
            except:
                embed_fea = embed_fea.data.numpy()
                targets = targets.data.numpy()

            plot_features.append(embed_fea)
            plot_labels.append(targets)
    plot_features = np.concatenate(plot_features, 0)
    plot_labels = np.concatenate(plot_labels, 0)
    net_dict = net.state_dict()
    centroids = net_dict['module.centroid'] if isinstance(net, nn.DataParallel) \
        else net_dict['self.centroid']
    try:
        centroids = centroids.data.cpu().numpy()
    except:
        centroids = centroids.data.numpy()
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C8', 'C9','C7']
    for label_idx in range(args.train_class_num):
        features = plot_features[plot_labels == label_idx, :]
        plt.scatter(
            features[:, 0],
            features[:, 1],
            c=colors[label_idx],
            s=1,
        )
    legends = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    legends = legends[:label_idx]
    print(f"max(plot_labels): {max(plot_labels)}")
    if max(plot_labels)!=args.train_class_num:
        features = plot_features[plot_labels == max(plot_labels), :]
        plt.scatter(
            features[:, 0],
            features[:, 1],
            c=colors[-1],  # we use gray C7 to denote unknown
            s=1,
        )
        legends.append('unknown')

    plt.scatter(
        centroids[0],
        centroids[1],
        c='black',
        marker="*",
        s=5,
    )
    legends.append('center')


    plt.legend(legends, loc='upper right')

    save_name = os.path.join(dirname, 'epoch_' + str(epoch) + '.png')
    plt.savefig(save_name, bbox_inches='tight', dpi=plot_quality)
    plt.close()


