
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torchvision
import numpy as np
import torchvision.transforms as transforms

import os
import argparse
import sys
#from models import *
sys.path.append("../..")
import backbones.cifar as models
from datasets import MNIST
from Utils import adjust_learning_rate, progress_bar, Logger, mkdir_p, Evaluation
from DFPLoss import DFPLoss
from DFPNet import DFPNet

model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Plotting')

# Dataset preperation
parser.add_argument('--train_class_num', default=7, type=int, help='Classes used in training')
parser.add_argument('--test_class_num', default=10, type=int, help='Classes used in testing')
parser.add_argument('--includes_all_train_class', default=True,  action='store_true',
                    help='If required all known classes included in testing')

# Others
parser.add_argument('--bs', default=256, type=int, help='batch size')
parser.add_argument('--evaluate', action='store_true', help='Evaluate without training')


# General MODEL parameters
parser.add_argument('--arch', default='LeNetPlus', choices=model_names, type=str, help='choosing network')
parser.add_argument('--embed_dim', default=2, type=int, help='embedding feature dimension')
parser.add_argument('--beta', default=1.0, type=float, help='wight of between-class distance loss')
parser.add_argument('--alpha', default=1.0, type=float, help='weight of total distance loss')
parser.add_argument('--distance', default='l2', choices=['l2','l1','dotproduct'],
                    type=str, help='choosing distance metric')
parser.add_argument('--scaled', default=True,  action='store_true',
                    help='If scale distance by sqrt(embed_dim)')


# Parameters for stage 1
parser.add_argument('--stage1_resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')

# Parameters for plotting
parser.add_argument('--plot_max', default=0, type=int, help='max examples to plot in each class, 0 indicates all.')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.plotter = './checkpoints/mnist/' + args.arch + '/plotter_%s_%s'%(args.alpha,args.beta)
if not os.path.isdir(args.plotter):
    mkdir_p(args.plotter)

print('==> Preparing data..')
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


trainset = MNIST(root='../../data', train=True, download=True, transform=transform,
                    train_class_num=args.train_class_num, test_class_num=args.test_class_num,
                    includes_all_train_class=args.includes_all_train_class)

testset = MNIST(root='../../data', train=False, download=True, transform=transform,
                   train_class_num=args.train_class_num, test_class_num=args.test_class_num,
                   includes_all_train_class=args.includes_all_train_class)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4)


assert os.path.isfile(args.stage1_resume)


def main():
    print(device)
    print('==> Building model..')
    net = DFPNet(backbone=args.arch, num_classes=args.train_class_num,
                 embed_dim=args.embed_dim, distance=args.distance, scaled=args.scaled)
    # # embed_dim = net.feat_dim if not args.embed_dim else args.embed_dim
    # criterion_cls = nn.CrossEntropyLoss()
    # criterion_dis = DFPLoss(beta=args.beta)
    # optimizer = optim.SGD(net.parameters(), lr=args.stage1_lr, momentum=0.9, weight_decay=5e-4)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.stage1_resume:
        # Load checkpoint.
        if os.path.isfile(args.stage1_resume):
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(args.stage1_resume)
            net.load_state_dict(checkpoint['net'])
            print("=> checkpoint loaded!")
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        print("Resume is required")
    plot_feature(net, trainloader, device, args.plotter, epoch=0, plot_class_num=args.train_class_num, maximum=args.plot_max)



def plot_feature(net, plotloader, device,dirname, epoch=0,plot_class_num=10, maximum=500):
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
    isinstance(net_dict, nn.DataParallel)
    for k, _ in net_dict.items():
        print(k)
    centroids = net_dict['module.centroids'] if isinstance(net_dict, nn.DataParallel) \
        else net_dict['centroids']
    print(centroids)
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for label_idx in range(plot_class_num):
        features = plot_features[plot_labels == label_idx,:]
        maximum = min(maximum, len(features)) if maximum>0 else len(features)
        plt.scatter(
            features[0:maximum, 0],
            features[0:maximum, 1],
            c=colors[label_idx],
            s=1,
        )
        # plt.scatter(
        #     centroids[label_idx, 0],
        #     centroids[label_idx, 1],
        #     c=colors[label_idx],
        #     marker='^',
        #     s=1.5,
        # )
    # currently only support 10 classes, for a good visualization.
    # change plot_class_num would lead to problems.
    legends= ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plt.legend(legends[0:plot_class_num], loc='upper right')
    save_name = os.path.join(dirname, 'epoch_' + str(epoch) + '.png')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()












if __name__ == '__main__':
    main()

