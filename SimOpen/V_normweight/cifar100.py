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
from datasets import CIFAR100
from Utils import adjust_learning_rate, progress_bar, Logger, mkdir_p, Evaluation
from DFPLoss import DFPLoss
from DFPNet import DFPNet
from MyPlotter import plot_feature
from energy_hist import energy_hist

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

# Dataset preperation
parser.add_argument('--train_class_num', default=50, type=int, help='Classes used in training')
parser.add_argument('--test_class_num', default=100, type=int, help='Classes used in testing')
parser.add_argument('--includes_all_train_class', default=True, action='store_true',
                    help='If required all known classes included in testing')

# General MODEL parameters
parser.add_argument('--arch', default='ResNet18', choices=model_names, type=str, help='choosing network')
parser.add_argument('--embed_dim', default=512, type=int, help='embedding feature dimension')

# Parameters for optimizer
parser.add_argument('--temperature', default=1, type=int, help='scaling cosine distance for exp')

# Parameters for stage 1 training
parser.add_argument('--stage1_resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--stage1_es', default=100, type=int, help='epoch size')
parser.add_argument('--stage1_lr', default=0.1, type=float, help='learning rate')  # works for MNIST
parser.add_argument('--stage1_lr_factor', default=0.1, type=float, help='learning rate Decay factor')  # works for MNIST
parser.add_argument('--stage1_lr_step', default=30, type=float, help='learning rate Decay step')  # works for MNIST
parser.add_argument('--stage1_bs', default=128, type=int, help='batch size')
parser.add_argument('--evaluate', action='store_true', help='Evaluate without training')

# Parameters for stage plotting
parser.add_argument('--plot', action='store_true', help='Plotting the training set.')
parser.add_argument('--plot_quality', default=200, type=int, help='DPI of plot figure')

# histogram figures for Energy model analysis
parser.add_argument('--hist_bins', default=100, type=int, help='divided into n bins')
parser.add_argument('--hist_norm', default=True, action='store_true', help='if norm the frequency to [0,1]')
parser.add_argument('--hist_save', action='store_true', help='if save the histogram figures')
parser.add_argument('--hist_list', default=["norm_fea","normweight_fea2cen","cosine_fea2cen"],
                    type=str, nargs='+', help='what outputs to analysis')


args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.checkpoint = './checkpoints/cifar100/%s-%s-%s-dim%s-T%s' % (
    args.train_class_num, args.test_class_num, args.arch, args.embed_dim,args.temperature)
if not os.path.isdir(args.checkpoint):
    mkdir_p(args.checkpoint)

# folder to save figures
args.plotfolder = os.path.join(args.checkpoint, "plotter")
if not os.path.isdir(args.plotfolder):
    mkdir_p(args.plotfolder)
# folder to save histogram
args.histfolder = os.path.join(args.checkpoint, "histogram")
if not os.path.isdir(args.histfolder):
    mkdir_p(args.histfolder)

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = CIFAR100(root='../../data', train=True, download=True, transform=transform_train,
                    train_class_num=args.train_class_num, test_class_num=args.test_class_num,
                    includes_all_train_class=args.includes_all_train_class)

testset = CIFAR100(root='../../data', train=False, download=True, transform=transform_test,
                   train_class_num=args.train_class_num, test_class_num=args.test_class_num,
                   includes_all_train_class=args.includes_all_train_class)

# data loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.stage1_bs, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.stage1_bs, shuffle=False, num_workers=4)


def main():
    print(device)
    stage1_dict = main_stage1()


def main_stage1():
    print(f"\nStart Stage-1 training ...\n")
    # for  initializing backbone, two branches, and centroids.
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Model
    print('==> Building model..')
    net = DFPNet(backbone=args.arch, num_classes=args.train_class_num, embed_dim=args.embed_dim)

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

    # after resume
    criterion = DFPLoss(temperature=args.temperature)
    optimizer = optim.SGD(net.parameters(), lr=args.stage1_lr, momentum=0.9, weight_decay=5e-4)
    if not args.evaluate:
        for epoch in range(start_epoch, args.stage1_es):
            adjust_learning_rate(optimizer, epoch, args.stage1_lr, factor=args.stage1_lr_factor, step=args.stage1_lr_step)
            print('\nStage_1 Epoch: %d | Learning rate: %f ' % (epoch + 1, optimizer.param_groups[0]['lr']))
            train_out = stage1_train(net, trainloader, optimizer, criterion, device)
            save_model(net, epoch, os.path.join(args.checkpoint, 'stage_1_last_model.pth'))
            logger.append([epoch + 1, train_out["train_loss"], train_out["accuracy"]])
            if args.plot:
                plot_feature(net, args, trainloader, device, args.plotfolder, epoch=epoch,
                             plot_class_num=args.train_class_num, plot_quality=args.plot_quality)
                plot_feature(net, args, testloader, device, args.plotfolder, epoch="test" + str(epoch),
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
    # loss_classification = 0
    # loss_distance = 0
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
        # loss_classification += (loss_dict['classification']).item()
        # loss_distance += (loss_dict['distance']).item()

        _, predicted = (out['normweight_fea2cen']).max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return {
        "train_loss": train_loss / (batch_idx + 1),
        # "loss_classification": loss_classification / (batch_idx + 1),
        # "loss_distance": loss_distance / (batch_idx + 1),
        "accuracy": correct / total
    }


def stage1_test(net, testloader, device):
    correct = 0
    total = 0
    norm_fea_list, normweight_fea2cen_list, cosine_fea2cen_list,softmax_list = [], [], [], []
    Target_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            out = net(inputs) # shape [batch,class]
            # energy = (out["normweight_fea2cen"]).sum(dim=1, keepdim=False)
            # energy = torch.logsumexp(out["normweight_fea2cen"], dim=1, keepdim=False)
            norm_fea_list.append(out["norm_fea"])
            normweight_fea2cen_list.append(out["normweight_fea2cen"])
            cosine_fea2cen_list.append(out["cosine_fea2cen"])
            softmax_list.append((out["cosine_fea2cen"].softmax(dim=1).max(dim=1,keepdim=False))[0])
            Target_list.append(targets)

            _, predicted = (out["normweight_fea2cen"]).max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), '| Acc: %.3f%% (%d/%d)'
                         % (100. * correct / total, correct, total))

    print("\nTesting results is {:.2f}%".format(100. * correct / total))

    norm_fea_list = torch.cat(norm_fea_list, dim=0)
    normweight_fea2cen_list = torch.cat(normweight_fea2cen_list, dim=0)
    cosine_fea2cen_list = torch.cat(cosine_fea2cen_list, dim=0)
    softmax_list = torch.cat(softmax_list, dim=0)
    Target_list = torch.cat(Target_list, dim=0)

    energy_hist(norm_fea_list, Target_list, args, "norm")
    energy_hist(torch.logsumexp(norm_fea_list, dim=1, keepdim=False), Target_list, args, "norm_energy")

    energy_hist(normweight_fea2cen_list, Target_list, args, "normweight")
    energy_hist(torch.logsumexp(normweight_fea2cen_list, dim=1, keepdim=False), Target_list, args, "normweight_energy")

    energy_hist(cosine_fea2cen_list, Target_list, args, "cosine")
    energy_hist(torch.logsumexp(cosine_fea2cen_list, dim=1, keepdim=False), Target_list, args, "cosine_energy")

    energy_hist(softmax_list, Target_list, args, "softmax")

    # # Energy analysis
    # Energy_list = torch.cat(Energy_list, dim=0)
    # Target_list = torch.cat(Target_list, dim=0)
    # unknown_label = Target_list.max()
    # unknown_Energy_list = Energy_list[Target_list == unknown_label]
    # known_Energy_list = Energy_list[Target_list != unknown_label]
    # unknown_hist = torch.histc(unknown_Energy_list, bins=args.hist_bins, min=Energy_list.min().data,
    #                            max=Energy_list.max().data)
    # known_hist = torch.histc(known_Energy_list, bins=args.hist_bins, min=Energy_list.min().data,
    #                            max=Energy_list.max().data)
    # print(f"unknown_hist: \n{unknown_hist}")
    # print(f"known_hist: \n{known_hist}")



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
