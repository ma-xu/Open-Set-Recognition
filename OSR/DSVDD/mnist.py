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
from DSVDDLoss import DSVDDLoss
from NetBuilder import NetBuilder
from MyPlotter import plot_feature
from helper import get_gap_stat

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))

# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
parser = argparse.ArgumentParser(description='PyTorch MNIST Training')

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

parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--epochs', default=50, type=int, help='epoch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')  # works for MNIST
parser.add_argument('--plot', action='store_true', help='Plotting the training set.')

parser.add_argument('--plot_quality', default=200, type=int, help='DPI of plot figure')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.checkpoint = './checkpoints/mnist/' + \
                  '/%s-%s-%s-%s' % (args.train_class_num, args.test_class_num, args.arch, args.embed_dim)
if not os.path.isdir(args.checkpoint):
    mkdir_p(args.checkpoint)

# folder to save figures
args.plotfolder = os.path.join(args.checkpoint, "plotter")
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
    print(f"------------Running on {device}------------")
    print('==> Building model..')
    net = NetBuilder(backbone=args.arch,  embed_dim=args.embed_dim)
    net = net.to(device)
    center = calcuate_center(net, trainloader, device)
    net._init_centroid(center)
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    if args.resume:
        # Load checkpoint.
        if os.path.isfile(args.stage1_resume):
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch']
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'Train Loss'])

    criterion = DSVDDLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr, step=20)
        print('\nStage_1 Epoch: %d | Learning rate: %f ' % (epoch + 1, optimizer.param_groups[0]['lr']))
        train_loss = train(net, trainloader, optimizer, criterion, device)
        save_model(net, epoch, os.path.join(args.checkpoint, 'last_model.pth'))
        logger.append([epoch + 1, train_loss])
        if args.plot:
            # plot training set
            plot_feature(net, args, trainloader, device, args.plotfolder, epoch=epoch, plot_quality=150)
            # plot testing set
            plot_feature(net, args, testloader, device, args.plotfolder, epoch='test_'+str(epoch), plot_quality=150)


# Training
def train(net, trainloader, optimizer, criterion, device):
    net.train()
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        out = net(inputs)
        loss = criterion(out)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f '
                     % (train_loss / (batch_idx + 1)))
    return train_loss


def calcuate_center(net, trainloader, device):
    feature = 0
    number = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            out = net(inputs)
            embed_feature = out["embed_fea"]
            batch_size = embed_feature.shape[0]
            feature += embed_feature.sum(dim=0)
            number += batch_size
    return feature/number


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
