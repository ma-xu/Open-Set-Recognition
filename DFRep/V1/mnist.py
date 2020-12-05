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
from MyPlotter import plot_feature, plot_distance
from helper import get_gap_stat

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
parser.add_argument('--similarity', default='cosine', choices=['l2', 'l1', 'cosine', 'dotproduct'],
                    type=str, help='choosing distance metric')

# Parameters for optimizer
parser.add_argument('--alpha', default=1.0, type=float, help='weight for similarity loss')
parser.add_argument('--temperature', default=1.0, type=float, help='softmax temperature')

# Parameters for stage 1
parser.add_argument('--stage1_resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--stage1_es', default=25, type=int, help='epoch size')
parser.add_argument('--stage1_lr', default=0.01, type=float, help='learning rate')  # works for MNIST

# Parameters for stage 2
parser.add_argument('--stage2_resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--stage2_es', default=25, type=int, help='epoch size')
parser.add_argument('--stage2_lr', default=0.001, type=float, help='learning rate')  # works for MNIST

# Parameters for stage plotting
parser.add_argument('--plot', action='store_true', help='Plotting the training set.')
parser.add_argument('--plot_max', default=0, type=int, help='max examples to plot in each class, 0 indicates all.')

parser.add_argument('--plot_quality', default=200, type=int, help='DPI of plot figure')
parser.add_argument('--bins', default=50, type=int, help='divided into n bins')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.checkpoint = './checkpoints/mnist/' + \
                  '/%s-%s_%s_%s-%s_%s_%s' % (args.train_class_num, args.test_class_num, args.arch, args.embed_dim)
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
    print(device)
    # stage1_dict = {
    #     'distance': {'thresholds': torch.ones(args.train_class_num)},
    #     'stat': None,
    #     'net': None
    # }
    #
    # if not args.evaluate and not os.path.isfile(args.stage2_resume):
    #     stage1_dict = main_stage1()
    # main_stage2(stage1_dict)
    stage1_dict = main_stage1()


def main_stage1():
    print(f"\nStart Stage-1 training ...\n")
    # for  initializing backbone, two branches, and centroids.
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Model
    print('==> Building model..')
    net = DFPNet(backbone=args.arch, num_classes=args.train_class_num, embed_dim=args.embed_dim,
                 similarity=args.similarity)

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
        logger.set_names(['Epoch', 'Train Loss', 'Classification Loss', 'Distance Loss', 'Train Acc.'])

    # after resume
    criterion = DFPLoss(alpha=args.alpha, temperature=args.temperature)
    optimizer = optim.SGD(net.parameters(), lr=args.stage1_lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(start_epoch, args.stage1_es):
        adjust_learning_rate(optimizer, epoch, args.stage1_lr, step=10)
        print('\nStage_1 Epoch: %d | Learning rate: %f ' % (epoch + 1, optimizer.param_groups[0]['lr']))
        train_out = stage1_train(net, trainloader, optimizer, criterion, device)
        save_model(net, epoch, os.path.join(args.checkpoint, 'stage_1_last_model.pth'))
        logger.append([epoch + 1, train_out["train_loss"], train_out["loss_classification"],
                       train_out["loss_distance"], train_out["accuracy"]])
        if args.plot:
            plot_feature(net, args, trainloader, device, args.plotfolder1, epoch=epoch,
                         plot_class_num=args.train_class_num, plot_quality=args.plot_quality)
            plot_feature(net, args, testloader, device, args.plotfolder1, epoch="test" + str(epoch),
                         plot_class_num=args.train_class_num + 1, plot_quality=args.plot_quality, testmode=True)

    logger.close()
    print(f"\nFinish Stage-1 training...\n")
    print("===> Evaluating ...")
    stage1_test(net, testloader, device)

    return {
        "net": net
    }


# Training
def stage1_train(net, trainloader, optimizer, criterion, device):
    net.train()
    train_loss = 0
    loss_classification = 0
    loss_distance = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        out = net(inputs)
        loss_dict = criterion(out, targets)
        loss = loss_dict['total']
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loss_classification += (loss_dict['classification']).item()
        loss_distance += (loss_dict['distance']).item()

        _, predicted = (out['sim_fea2cen']).max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return {
        "train_loss": train_loss / (batch_idx + 1),
        "loss_classification": loss_classification / (batch_idx + 1),
        "loss_distance": loss_distance / (batch_idx + 1),
        "accuracy": correct / total
    }


def stage1_test(net, testloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            out = net(inputs)
            _, predicted = (out["sim_fea2cen"]).max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), '| Acc: %.3f%% (%d/%d)'
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
