from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import numpy as np
import torchvision.transforms as transforms

import os
import argparse
import sys

# from models import *
sys.path.append("../..")
import backbones.cifar as models
from datasets import MNIST
from Utils import adjust_learning_rate, progress_bar, Logger, mkdir_p, Evaluation
from DFPLoss import DFPLoss
from DFPNet import DFPNet
from MyPlotter import plot_feature, plot_distance,plot_similarity
from Generater import generater_input, generater_unknown

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

# Dataset preperation
parser.add_argument('--train_class_num', default=7, type=int, help='Classes used in training')
parser.add_argument('--test_class_num', default=10, type=int, help='Classes used in testing')
parser.add_argument('--includes_all_train_class', default=True, action='store_true',
                    help='If required all known classes included in testing')

# Others
parser.add_argument('--bs', default=256, type=int, help='batch size')
parser.add_argument('--evaluate', action='store_true', help='Evaluate without training')

# General MODEL parameters
parser.add_argument('--arch', default='LeNetPlus', choices=model_names, type=str, help='choosing network')
parser.add_argument('--embed_dim', default=2, type=int, help='embedding feature dimension')
parser.add_argument('--distance', default='cosine', choices=['l2', 'l1', 'cosine','dotproduct'],
                    type=str, help='choosing distance metric')
parser.add_argument('--scaled', default=True, action='store_true',
                    help='If scale distance by sqrt(embed_dim)')
parser.add_argument('--norm_centroid', action='store_true', help='If nomalize the centroid for calculation similarities')
parser.add_argument('--p_value', default=0.01, type=float, help='default statistical p_value threshold,'
                                                                ' usually 0.05. 0.01')

# Parameters for stage 1
parser.add_argument('--stage1_resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--stage1_es', default=35, type=int, help='epoch size')
parser.add_argument('--stage1_lr', default=0.01, type=float, help='learning rate')  # works for MNIST

# Parameters for stage 2
parser.add_argument('--stage2_resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--stage2_es', default=50, type=int, help='epoch size')
parser.add_argument('--stage2_lr', default=0.001, type=float, help='learning rate')  # works for MNIST

# Parameters for stage plotting
parser.add_argument('--plot', action='store_true', help='Plotting the training set.')
parser.add_argument('--plot_max', default=0, type=int, help='max examples to plot in each class, 0 indicates all.')


parser.add_argument('--plot_quality', default=200, type=int, help='DPI of plot figure')
parser.add_argument('--bins', default=50, type=int, help='divided into n bins')
parser.add_argument('--tail_number', default=50, type=int,
                    help='number of maximum distance we do not take into account, '
                         'which may be anomaly or wrong labeled.')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.checkpoint = './checkpoints/mnist/' + args.arch + \
                  '/embed%s_%s_norm%s' % (args.embed_dim, args.distance, args.norm_centroid)
if not os.path.isdir(args.checkpoint):
    mkdir_p(args.checkpoint)

# folder to save figures
args.plotfolder1 = os.path.join(args.checkpoint, "plotter_Stage1")
if not os.path.isdir(args.plotfolder1):
    mkdir_p(args.plotfolder1)
# folder to save figures
# args.plotfolder2 = os.path.join(args.checkpoint, "plotter_Stage2")
# if not os.path.isdir(args.plotfolder2):
#     mkdir_p(args.plotfolder2)

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

# data loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4)


def main():
    print(device)

    stage1_dict = main_stage1()
    # main_stage2(stage1_dict)

    #     centroids = cal_centroids(net1, device)
    # main_stage2(net1, centroids)


def main_stage1():
    print(f"\nStart Stage-1 training ...\n")
    # for  initializing backbone, two branches, and centroids.
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Model
    print('==> Building model..')
    net = DFPNet(backbone=args.arch, num_classes=args.train_class_num, embed_dim=args.embed_dim,
                 distance=args.distance, scaled=args.scaled)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.stage1_lr, momentum=0.9, weight_decay=5e-4)

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
            start_epoch = checkpoint['epoch']
            logger = Logger(os.path.join(args.checkpoint, 'log_stage1.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log_stage1.txt'))
        logger.set_names(['Epoch', 'Train Loss', 'Train Acc.'])

    if not args.evaluate:
        for epoch in range(start_epoch, args.stage1_es):
            adjust_learning_rate(optimizer, epoch, args.stage1_lr, step=10)
            print('\nStage_1 Epoch: %d | Learning rate: %f ' % (epoch + 1, optimizer.param_groups[0]['lr']))
            train_out = stage1_train(net, trainloader, optimizer, criterion, device)
            save_model(net, epoch, os.path.join(args.checkpoint, 'stage_1_last_model.pth'))
            logger.append([epoch + 1, train_out["train_loss"], train_out["accuracy"]])
            if args.plot:
                plot_feature(net, trainloader, device, args.plotfolder1, epoch=epoch,
                             plot_class_num=args.train_class_num, maximum=args.plot_max,
                             plot_quality=args.plot_quality, norm_centroid=args.norm_centroid)
    if args.plot:
        # plot the test set
        plot_feature(net, testloader, device, args.plotfolder1, epoch="test",
                     plot_class_num=args.train_class_num + 1, maximum=args.plot_max,
                     plot_quality=args.plot_quality, norm_centroid=args.norm_centroid)

    # calculating distances for last epoch
    similarity_results = plot_similarity(net, trainloader, device, args)

    logger.close()
    print(f"\nFinish Stage-1 training...\n")
    print("===> Evaluating ...")
    stage1_test(net, testloader, device)

    return {"net": net,
            "distance": similarity_results
            }


# Training
def stage1_train(net, trainloader, optimizer, criterion, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        out = net(inputs)
        loss = criterion(out['sim_fea2cen'], targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = (out['sim_fea2cen']).max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return {
        "train_loss": train_loss / (batch_idx + 1),
        "accuracy": correct / total
    }


def stage1_test(net, testloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            out = net(inputs)
            _, predicted = (out["logits"]).max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), '| Acc: %.3f%% (%d/%d)'
                         % (100. * correct / total, correct, total))

    print("\nTesting results is {:.2f}%".format(100. * correct / total))


def save_model(net, epoch, path, **kwargs):
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
    }
    for key, value in kwargs.items():
        state[key] = value
    torch.save(state, path)


if __name__ == '__main__':
    main()
