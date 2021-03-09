from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as functional
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
from datasets import CIFAR10
from Utils import adjust_learning_rate, progress_bar, Logger, mkdir_p, Evaluation, save_model
from Losses import CenterLoss, SoftmaxLoss, ArcFaceLoss, NormFaceLoss, PSoftmaxLoss
from BuildNet import BuildNet
from Distance import Distance

# loss: ->  "SoftmaxLoss",  "PSoftmaxLoss"
# -> "possibility",'norm','energy'
# python3 cifar10.py --loss SoftmaxLoss --train_class_num 5
# python3 cifar10.py --loss PSoftmaxLoss --test_class_num 5

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

# Dataset preperation
parser.add_argument('--train_class_num', default=5, type=int, help='Classes used in training')
parser.add_argument('--test_class_num', default=10, type=int, help='Classes used in testing')
parser.add_argument('--includes_all_train_class', default=True, action='store_true',
                    help='If required all known classes included in testing')

# General MODEL parameters
parser.add_argument('--arch', default='ResNet18', choices=model_names, type=str, help='choosing network')
parser.add_argument('--embed_dim', default=128, type=int, help='embedding feature dimension')
parser.add_argument('--loss', default='SoftmaxLoss',
                    choices=["SoftmaxLoss", "PSoftmaxLoss"],
                    type=str, help='choosing network')
# parser.add_argument('--openmetric', default='possibility',
#                     choices=["possibility", "distance", 'norm', 'energy', 'cosine'],
#                     type=str, help='choosing network')

# Parameters for losses
parser.add_argument('--centerloss_weight', default=0.03, type=float, help='center loss')
parser.add_argument('--scaling', default=16, type=float, help='center loss')
parser.add_argument('--m', default=0.5, type=float, help='center loss')

# Parameters for training
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--es', default=100, type=int, help='epoch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr_factor', default=0.1, type=float, help='learning rate Decay factor')  # works for MNIST
parser.add_argument('--lr_step', default=30, type=float, help='learning rate Decay step')  # works for MNIST
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--evaluate', action='store_true', help='Evaluate without training')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.checkpoint = './checkpoints/cifar10/%s_%s_%s_%s_dim%s-c%s-s%s-m%s' % (
    args.loss, args.train_class_num, args.test_class_num, args.arch, args.embed_dim,
    args.centerloss_weight, args.scaling, args.m)
if not os.path.isdir(args.checkpoint):
    mkdir_p(args.checkpoint)

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
trainset = CIFAR10(root='../../data', train=True, download=True, transform=transform_train,
                   train_class_num=args.train_class_num, test_class_num=args.test_class_num,
                   includes_all_train_class=args.includes_all_train_class)

# data loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=4)


loss_Dict = {"CenterLoss": CenterLoss(centerloss_weight=args.centerloss_weight, num_classes=args.train_class_num),
             "SoftmaxLoss": SoftmaxLoss(),
             "ArcFaceLoss": ArcFaceLoss(scaling=args.scaling, m=args.m),
             "NormFaceLoss": NormFaceLoss(scaling=args.scaling),
             "PSoftmaxLoss": PSoftmaxLoss()}
criterion = loss_Dict[args.loss]
criterion = criterion.to(device)


def main():
    print(f"\nStart  training ...\n")
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    print('==> Building model..')
    net = BuildNet(backbone=args.arch, num_classes=args.train_class_num, embed_dim=args.embed_dim)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    if args.resume:
        # Load checkpoint.
        if os.path.isfile(args.resume):
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            loggerList = []
            for i in range(args.train_class_num, args.test_class_num):
                loggerList.append( Logger(os.path.join(args.checkpoint, f'log{i}.txt'), resume=True))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        loggerList = []
        for i in range(args.train_class_num, args.test_class_num):
            logger = Logger(os.path.join(args.checkpoint, f'log{i}.txt'))
            logger.set_names(['Epoch', 'Train Loss', 'Train Acc.', "Pos-F1", 'Norm-F1', 'Energy-F1'])
            loggerList.append(logger)

    if not args.evaluate:
        for epoch in range(start_epoch, args.es):
            adjust_learning_rate(optimizer, epoch, args.lr,
                                 factor=args.lr_factor, step=args.lr_step)
            print('\nEpoch: %d | Learning rate: %f ' % (epoch + 1, optimizer.param_groups[0]['lr']))
            train_out = train(net, trainloader, optimizer, criterion, device)
            save_model(net, optimizer, epoch, os.path.join(args.checkpoint, 'last_model.pth'))

            for test_class_num in range(args.train_class_num, args.test_class_num):
                testset = CIFAR10(root='../../data', train=False, download=True, transform=transform_test,
                                  train_class_num=args.train_class_num, test_class_num=test_class_num,
                                  includes_all_train_class=args.includes_all_train_class)
                testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4)
                test_out = test(net, testloader, criterion, device)
                logger = loggerList[test_class_num-args.train_class_num]
                logger.append([epoch + 1, train_out["train_loss"], train_out["accuracy"],
                           test_out["best_F1_possibility"],test_out["best_F1_norm"], test_out["best_F1_energy"]])
        logger.close()
        print(f"\nFinish training...\n")



def train(net, trainloader, optimizer, criterion, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        out = net(inputs)
        loss_dict = criterion(out, targets)
        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # "CenterLoss", "SoftmaxLoss", "ArcFaceLoss", "NormFaceLoss", "PSoftmaxLoss"
        if args.loss in ["CenterLoss", "SoftmaxLoss", ]:
            _, predicted = (out['dotproduct_fea2cen']).max(1)
        elif args.loss in ["ArcFaceLoss", "NormFaceLoss", ]:
            _, predicted = (out['cosine_fea2cen']).max(1)
        else:  # PSoftmaxLoss
            _, predicted = (out['normweight_fea2cen']).max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return {
        "train_loss": train_loss / (batch_idx + 1),
        "accuracy": correct / total
    }


def test(net, testloader, criterion, device, intervals=20):
    normfea_list = []  # extracted feature norm
    cosine_list = []  # extracted cosine similarity
    energy_list = []  # energy value
    embed_fea_list = []
    dotproduct_fea2cen_list = []
    normweight_fea2cen_list = []
    Target_list = []
    Predict_list = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            out = net(inputs)  # shape [batch,class]

            # Test cannot calculate loss beacuse of class number dismatch.
            # loss_dict = criterion(out, targets)
            # loss = loss_dict['loss']
            # test_loss += loss.item()

            normfea_list.append(out["norm_fea"])
            cosine_list.append(out["cosine_fea2cen"])
            energy_list.append(out["energy"])
            embed_fea_list.append(out["embed_fea"])
            dotproduct_fea2cen_list.append(out["dotproduct_fea2cen"])
            normweight_fea2cen_list.append(out["normweight_fea2cen"])
            Target_list.append(targets)

            # "CenterLoss", "SoftmaxLoss", "ArcFaceLoss", "NormFaceLoss", "PSoftmaxLoss"
            if args.loss in ["CenterLoss", "SoftmaxLoss", ]:
                _, predicted = (out['dotproduct_fea2cen']).max(1)
            elif args.loss in ["ArcFaceLoss", "NormFaceLoss", ]:
                _, predicted = (out['cosine_fea2cen']).max(1)
            else:  # PSoftmaxLoss
                _, predicted = (out['normweight_fea2cen']).max(1)
            Predict_list.append(predicted)

            progress_bar(batch_idx, len(testloader), "|||")

    normfea_list = torch.cat(normfea_list, dim=0)
    cosine_list = torch.cat(cosine_list, dim=0)
    energy_list = torch.cat(energy_list, dim=0)
    embed_fea_list = torch.cat(embed_fea_list, dim=0)
    dotproduct_fea2cen_list = torch.cat(dotproduct_fea2cen_list, dim=0)
    normweight_fea2cen_list = torch.cat(normweight_fea2cen_list, dim=0)
    Target_list = torch.cat(Target_list, dim=0)
    Predict_list = torch.cat(Predict_list, dim=0)

    # "CenterLoss", "SoftmaxLoss", "ArcFaceLoss", "NormFaceLoss", "PSoftmaxLoss"
    # "possibility", "distance",'norm','energy','cosine'
    best_F1_possibility = 0
    best_F1_norm = 0
    best_F1_energy = 0

    # for these unbounded metric, we explore more intervals by *5 to achieve a relatively fair comparison.
    expand_factor = 5
    Predict_list_possibility = Predict_list.copy_()
    Predict_list_norm = Predict_list.copy_()
    Predict_list_energy = Predict_list.copy_()

    # possibility
    if args.loss in ["SoftmaxLoss"]:
        openmetric_possibility = dotproduct_fea2cen_list
    if args.loss in ["PSoftmaxLoss"]:
        openmetric_possibility = normweight_fea2cen_list
    openmetric_possibility, _ = torch.softmax(openmetric_possibility, dim=1).max(dim=1)
    for thres in np.linspace(0.0, 1.0, intervals):
        Predict_list_possibility[openmetric_possibility < thres] = args.train_class_num
        eval = Evaluation(Predict_list_possibility.cpu().numpy(), Target_list.cpu().numpy())
        if eval.f1_measure > best_F1_possibility:
            best_F1_possibility = eval.f1_measure

    # norm
    openmetric_norm = normfea_list.squeeze(dim=1)
    threshold_min_norm = openmetric_norm.min().item()
    threshold_max_norm = openmetric_norm.max().item()
    for thres in np.linspace(threshold_min_norm, threshold_max_norm, expand_factor * intervals):
        Predict_list_norm[openmetric_norm < thres] = args.train_class_num
        eval = Evaluation(Predict_list_norm.cpu().numpy(), Target_list.cpu().numpy())
        if eval.f1_measure > best_F1_norm:
            best_F1_norm = eval.f1_measure

    # energy
    openmetric_energy = energy_list
    threshold_min_energy = openmetric_energy.min().item()
    threshold_max_energy = openmetric_energy.max().item()
    for thres in np.linspace(threshold_min_energy, threshold_max_energy, expand_factor * intervals):
        Predict_list_energy[openmetric_energy < thres] = args.train_class_num
        eval = Evaluation(Predict_list_energy.cpu().numpy(), Target_list.cpu().numpy())
        if eval.f1_measure > best_F1_energy:
            best_F1_energy = eval.f1_measure

    print(f"Best Possibility F1 is: {best_F1_possibility} | Norm F1 is :{best_F1_norm} | Energy F1 is: {best_F1_energy}")
    return {
        "best_F1_possibility": best_F1_possibility,
        "best_F1_norm": best_F1_norm,
        "best_F1_energy": best_F1_energy
    }


if __name__ == '__main__':
    main()
