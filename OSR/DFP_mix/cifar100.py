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
from DFPLoss import DFPLoss, DFPLossGeneral
from DFPNet import DFPNet
from MyPlotter import plot_feature, plot_distance

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

# Others
parser.add_argument('--bs', default=256, type=int, help='batch size')
parser.add_argument('--evaluate', action='store_true', help='Evaluate without training')

# General MODEL parameters
parser.add_argument('--arch', default='ResNet18', choices=model_names, type=str, help='choosing network')
parser.add_argument('--embed_dim', default=64, type=int, help='embedding feature dimension')
# parser.add_argument('--embed_reduction', default=8, type=int, help='reduction ratio for embedding like SENet.')
# alpha is deprecated.
parser.add_argument('--alpha', default=1.0, type=float, help='weight of total distance loss')
parser.add_argument('--beta', default=1.0, type=float, help='wight of between-class distance loss')
parser.add_argument('--sigma', default=1.0, type=float, help='wight of center-to-center distance loss')
parser.add_argument('--gamma', default=1.0, type=float, help='wight of generated data distance loss')
parser.add_argument('--distance', default='cosine', choices=['l2', 'l1', 'cosine'],
                    type=str, help='choosing distance metric')
parser.add_argument('--scaled', default=True, action='store_true',
                    help='If scale distance by sqrt(embed_dim)')
parser.add_argument('--cosine_weight', default=1.0, type=float, help='wight of magnifying the cosine distance')
parser.add_argument('--p_value', default=0.01, type=float, help='default statistical p_value threshold,'
                                                                ' usually 0.05. 0.01')

# Parameters for stage 1
parser.add_argument('--stage1_resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--stage1_es', default=100, type=int, help='epoch size')
parser.add_argument('--stage1_lr', default=0.1, type=float, help='learning rate') # works for MNIST

# Parameters for stage 2
parser.add_argument('--stage2_resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--stage2_es', default=50, type=int, help='epoch size')
parser.add_argument('--stage2_lr', default=0.01, type=float, help='learning rate') # works for MNIST


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
args.checkpoint = './checkpoints/cifar/' + args.arch +\
                  '/Alpha%s_Beta%s_Sigma%s_Cosine%s_Embed%s' % (
                      args.alpha, args.beta,args.sigma,args.cosine_weight,args.embed_dim)
if not os.path.isdir(args.checkpoint):
    mkdir_p(args.checkpoint)

# folder to save figures
if args.plot:
    args.plotfolder1 = os.path.join(args.checkpoint,"plotter_Stage1")
    if not os.path.isdir(args.plotfolder1):
        mkdir_p(args.plotfolder1)
    # folder to save figures
    args.plotfolder2 = os.path.join(args.checkpoint,"plotter_Stage2")
    if not os.path.isdir(args.plotfolder2):
        mkdir_p(args.plotfolder2)

print('==> Preparing data..')
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

trainset = CIFAR100(root='../../data', train=True, download=True, transform=transform,
                    train_class_num=args.train_class_num, test_class_num=args.test_class_num,
                    includes_all_train_class=args.includes_all_train_class)

testset = CIFAR100(root='../../data', train=False, download=True, transform=transform,
                   train_class_num=args.train_class_num, test_class_num=args.test_class_num,
                   includes_all_train_class=args.includes_all_train_class)

# data loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4)


def main():
    print(device)

    stage1_dict = main_stage1()
    main_stage2(stage1_dict)

    #     centroids = cal_centroids(net1, device)
    # main_stage2(net1, centroids)


def main_stage1():
    print(f"\nStart Stage-1 training ...\n")
    # for  initializing backbone, two branches, and centroids.
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Model
    print('==> Building model..')
    net = DFPNet(backbone=args.arch, num_classes=args.train_class_num, embed_dim=args.embed_dim,
                 distance=args.distance, scaled=args.scaled, cosine_weight=args.cosine_weight)
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
        logger = Logger(os.path.join(args.checkpoint, 'log_stage1.txt'))
        logger.set_names(['Epoch', 'Train Loss', 'Softmax Loss', 'Distance Loss',
                          'Within Loss', 'Between Loss', 'Cen2cen Loss', 'Train Acc.'])

    if not args.evaluate:
        for epoch in range(start_epoch, args.stage1_es):
            adjust_learning_rate(optimizer, epoch, args.stage1_lr, step=30)
            print('\nStage_1 Epoch: %d | Learning rate: %f ' % (epoch + 1, optimizer.param_groups[0]['lr']))
            train_out = stage1_train(net, trainloader, optimizer, criterion_dis, device)
            save_model(net, epoch, os.path.join(args.checkpoint,'stage_1_last_model.pth'))
            # ['Epoch', 'Train Loss', 'Softmax Loss', 'Distance Loss',
            # 'Within Loss', 'Between Loss','Cen2cen loss', 'Train Acc.']
            logger.append([epoch + 1, train_out["train_loss"], 0.0,
                           train_out["dis_loss_total"], train_out["dis_loss_within"],
                           train_out["dis_loss_between"], train_out["dis_loss_cen2cen"], train_out["accuracy"]])
            if args.plot:
                plot_feature(net, trainloader, device, args.plotfolder1, epoch=epoch,
                             plot_class_num=args.train_class_num, maximum=args.plot_max,plot_quality=args.plot_quality)
    if args.plot:
        # plot the test set
        plot_feature(net, testloader, device, args.plotfolder1, epoch="test",
                     plot_class_num=args.train_class_num + 1, maximum=args.plot_max, plot_quality=args.plot_quality)

    # calculating distances for last epoch
    distance_results = plot_distance(net, trainloader, device, args)

    logger.close()
    print(f"\nFinish Stage-1 training...\n")
    print("===> Evaluating ...")
    stage1_test(net, testloader, device)

    return {"net": net,
            "distance": distance_results
            }


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
        loss_dis = criterion_dis(out, targets)
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

        _, predicted = (out["dist_fea2cen"]).min(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return {
        "train_loss": train_loss / (batch_idx + 1),
        "dis_loss_total": dis_loss_total / (batch_idx + 1),
        "dis_loss_within": dis_loss_within / (batch_idx + 1),
        "dis_loss_between": dis_loss_between / (batch_idx + 1),
        "dis_loss_cen2cen": dis_loss_cen2cen / (batch_idx + 1),
        "accuracy": correct / total
    }


def stage1_test(net,testloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            out = net(inputs)
            _, predicted = (out["dist_fea2cen"]).min(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), '| Acc: %.3f%% (%d/%d)'
                         % ( 100. * correct / total, correct, total))

    print("\nTesting results is {:.2f}%".format(100. * correct / total))


def main_stage2(stage1_dict):
    net1 = stage1_dict["net"]
    thresholds = stage1_dict["distance"]["thresholds"]

    print(f"\n===> Start Stage-2 training...\n")
    start_epoch = 0
    print('==> Building model..')
    # net2 = DFPNet(backbone=args.arch, num_classes=args.train_class_num, embed_dim=args.embed_dim,
    #              distance=args.distance, scaled=args.scaled, cosine_weight=args.cosine_weight,thresholds=thresholds)
    net2 = DFPNet(backbone=args.arch, num_classes=args.train_class_num, embed_dim=args.embed_dim,
                  distance=args.distance, scaled=args.scaled, cosine_weight=args.cosine_weight, thresholds=thresholds)
    net2 = net2.to(device)

    criterion_dis = DFPLossGeneral(beta=args.beta, sigma=args.sigma,gamma=args.gamma)
    optimizer = optim.SGD(net2.parameters(), lr=args.stage2_lr, momentum=0.9, weight_decay=5e-4)

    if not args.evaluate:
        init_stage2_model(net1, net2)
    if device == 'cuda':
        net2 = torch.nn.DataParallel(net2)
        cudnn.benchmark = True

    if args.stage2_resume:
        # Load checkpoint.
        if os.path.isfile(args.stage2_resume):
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(args.stage1_resume)
            net2.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch']
            logger = Logger(os.path.join(args.checkpoint, 'log_stage2.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log_stage2.txt'))
        logger.set_names(['Epoch', 'Train Loss', 'Within Loss', 'Between Loss', 'Within-Gen Loss', 'Between-Gen Loss',
                          'Random loss', 'Train Acc.'])

    if not args.evaluate:
        for epoch in range(start_epoch, args.stage2_es):
            adjust_learning_rate(optimizer, epoch, args.stage2_lr, step=20)
            print('\nStage_2 Epoch: %d | Learning rate: %f ' % (epoch + 1, optimizer.param_groups[0]['lr']))
            train_out = stage2_train(net2, trainloader, optimizer, criterion_dis, device)
            save_model(net2, epoch, os.path.join(args.checkpoint, 'stage_2_last_model.pth'))
            # ['Epoch', 'Train Loss', 'Softmax Loss', 'Distance Loss',
            # 'Within Loss', 'Between Loss','Cen2cen loss', 'Train Acc.']
            logger.append([epoch + 1, train_out["dis_loss_total"], train_out["dis_loss_within"],
                           train_out["dis_loss_between"],train_out["dis_loss_within_gen"],
                           train_out["dis_loss_between_gen"], train_out["dis_loss_cen2cen"], train_out["accuracy"]])
            if args.plot:
                plot_feature(net2, trainloader, device, args.plotfolder2, epoch=epoch,
                             plot_class_num=args.train_class_num, maximum=args.plot_max, plot_quality=args.plot_quality)
    if args.plot:
        # plot the test set
        plot_feature(net2, testloader, device, args.plotfolder2, epoch="test",
                     plot_class_num=args.train_class_num + 1, maximum=args.plot_max, plot_quality=args.plot_quality)

    # calculating distances for last epoch
    # distance_results = plot_distance(net2, trainloader, device, args)

    logger.close()
    print(f"\nFinish Stage-2 training...\n")
    print("===> Evaluating ...")
    stage1_test(net2, testloader, device)
    return net2

# Training
def stage2_train(net, trainloader, optimizer, criterion_dis, device):
    net.train()
    train_loss = 0
    cls_loss = 0
    dis_loss_total = 0
    dis_loss_within = 0
    dis_loss_between = 0
    dis_loss_within_gen = 0
    dis_loss_between_gen = 0
    dis_loss_cen2cen = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        out = net(inputs)
        # loss_cls = criterion_cls(out["logits"], targets)
        loss_dis = criterion_dis(out, targets)
        # loss = loss_cls + args.alpha * (loss_dis["total"])
        loss = args.alpha * (loss_dis["total"])
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        cls_loss += 0
        dis_loss_total += loss_dis["total"].item()
        dis_loss_within += loss_dis["within"].item()
        dis_loss_between += loss_dis["between"].item()
        dis_loss_within_gen += loss_dis["within_gen"].item()
        dis_loss_between_gen += loss_dis["between_gen"].item()
        dis_loss_cen2cen += loss_dis["cen2cen"].item()

        _, predicted = (out["dist_fea2cen"]).min(1)
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
        "dis_loss_within_gen": dis_loss_within_gen / (batch_idx + 1),
        "dis_loss_between_gen": dis_loss_between_gen / (batch_idx + 1),
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


def init_stage2_model(net1, net2):
    # net1: net from stage 1.
    # net2: net from stage 2.
    dict1 = net1.state_dict()
    dict2 = net2.state_dict()
    for k, v in dict1.items():
        if k.startswith("module.1."):
            k = k[9:]   # remove module.1.
        if k.startswith("module."):
            k = k[7:]   # remove module.1.
        dict2[k] = v
    net2.load_state_dict(dict2)

if __name__ == '__main__':
    main()
