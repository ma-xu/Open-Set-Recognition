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
from MyPlotter import plot_feature

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
# parser.add_argument('--embed_reduction', default=8, type=int, help='reduction ratio for embedding like SENet.')
# alpha is deprecated.
parser.add_argument('--alpha', default=1.0, type=float, help='weight of total distance loss')
parser.add_argument('--beta', default=1.0, type=float, help='wight of between-class distance loss')
parser.add_argument('--sigma', default=1.0, type=float, help='wight of center-to-center distance loss')
parser.add_argument('--distance', default='cosine', choices=['l2', 'l1', 'cosine'],
                    type=str, help='choosing distance metric')
parser.add_argument('--scaled', default=True, action='store_true',
                    help='If scale distance by sqrt(embed_dim)')

# Parameters for stage 1
parser.add_argument('--stage1_resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--stage1_es', default=100, type=int, help='epoch size')
parser.add_argument('--stage1_lr', default=0.01, type=float, help='learning rate') # works for MNIST


# Parameters for stage plotting
parser.add_argument('--plot', default=True, action='store_true', help='Plotting the training set.')
parser.add_argument('--plot_max', default=0, type=int, help='max examples to plot in each class, 0 indicates all.')
parser.add_argument('--plot_quality', default=200, type=int, help='DPI of plot figure')


args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.checkpoint = './checkpoints/mnist/' + args.arch
if not os.path.isdir(args.checkpoint):
    mkdir_p(args.checkpoint)

# folder to save figures
args.plotfolder = './checkpoints/mnist/' + args.arch + '/plotter_%s_%s_%s' % (args.alpha, args.beta,args.sigma)
if not os.path.isdir(args.plotfolder):
    mkdir_p(args.plotfolder)

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
    net1, centroids = None, None
    if not args.evaluate:
        net1 = main_stage1()
    #     centroids = cal_centroids(net1, device)
    # main_stage2(net1, centroids)


def main_stage1():
    print(f"\nStart Stage-1 training ...\n")
    # for  initializing backbone, two branches, and centroids.
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Model
    print('==> Building model..')
    net = DFPNet(backbone=args.arch, num_classes=args.train_class_num,
                 embed_dim=args.embed_dim, distance=args.distance, scaled=args.scaled)
    # embed_dim = net.feat_dim if not args.embed_dim else args.embed_dim
    # criterion_cls = nn.CrossEntropyLoss()
    criterion_dis = DFPLoss(beta=args.beta, sigma=args.sigma)
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
        logger = Logger(os.path.join(args.checkpoint, 'log_stage1_%s_%s_%s.txt' % (args.alpha, args.beta.args.sigma)))
        logger.set_names(['Epoch', 'Train Loss', 'Softmax Loss', 'Distance Loss',
                          'Within Loss', 'Between Loss', 'Cen2cen Loss', 'Train Acc.'])

    for epoch in range(start_epoch, start_epoch + args.stage1_es):
        print('\nStage_1 Epoch: %d | Learning rate: %f ' % (epoch + 1, optimizer.param_groups[0]['lr']))
        adjust_learning_rate(optimizer, epoch, args.stage1_lr, step=30)
        train_out = stage1_train(net, trainloader, optimizer, criterion_dis, device)
        save_model(net, epoch, os.path.join(args.checkpoint,
                                            'stage_1_last_model_%s_%s_%s.pth' % (args.alpha, args.beta, args.sigma)))
        # ['Epoch', 'Train Loss', 'Softmax Loss', 'Distance Loss',
        # 'Within Loss', 'Between Loss','Cen2cen loss', 'Train Acc.']
        logger.append([epoch + 1, train_out["train_loss"], '-',
                       train_out["dis_loss_total"], train_out["dis_loss_within"],
                       train_out["dis_loss_between"], train_out["dis_loss_cen2cen"], train_out["accuracy"]])
        if args.plot:
            plot_feature(net, trainloader, device, args.plotfolder, epoch=epoch,
                         plot_class_num=args.train_class_num, maximum=args.plot_max,plot_quality=args.plot_quality)
    logger.close()
    print(f"\nFinish Stage-1 training...\n")
    return net


# Training
def stage1_train(net, trainloader, optimizer, criterion_dis, device):
    net.train()
    train_loss = 0
    cls_loss = 0
    dis_loss_total = 0
    dis_loss_within = 0
    dis_loss_between = 0
    dis_loss_cen2cen = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        out = net(inputs)
        # loss_cls = criterion_cls(out["logits"], targets)
        loss_dis = criterion_dis(out["dist_fea2cen"], out["dist_cen2cen"], targets)
        # loss = loss_cls + args.alpha * (loss_dis["total"])
        loss = args.alpha * (loss_dis["total"])
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        cls_loss += 0
        dis_loss_total += loss_dis["total"].item()
        dis_loss_within += loss_dis["within"].item()
        dis_loss_between += loss_dis["between"].item()
        dis_loss_cen2cen += loss_dis["cen2cen"].item()

        _, predicted = (out["dist_fea2cen"]).max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return {
        "train_loss": train_loss / (batch_idx + 1),
        # "cls_loss": cls_loss / (batch_idx + 1),
        "dis_loss_total": dis_loss_total / (batch_idx + 1),
        "dis_loss_within": dis_loss_within / (batch_idx + 1),
        "dis_loss_between": dis_loss_between / (batch_idx + 1),
        "dis_loss_cen2cen": dis_loss_cen2cen / (batch_idx + 1),
        "accuracy": correct / total
    }


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
