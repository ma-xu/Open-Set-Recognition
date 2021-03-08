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

# loss: -> "CenterLoss", "SoftmaxLoss", "ArcFaceLoss", "NormFaceLoss", "PSoftmaxLoss"
# openmetric: -> "possibility", "distance",'norm','energy','cosine'
# python3 cifar10.py --loss SoftmaxLoss --openmetric possibility
# python3 cifar10.py --loss CenterLoss --openmetric possibility
# python3 cifar10.py --loss CenterLoss --openmetric distance
# python3 cifar10.py --loss ArcFaceLoss --openmetric possibility
# python3 cifar10.py --loss ArcFaceLoss --openmetric cosine
# python3 cifar10.py --loss NormFaceLoss --openmetric possibility
# python3 cifar10.py --loss NormFaceLoss --openmetric cosine
# python3 cifar10.py --loss PSoftmaxLoss --openmetric possibility
# python3 cifar10.py --loss PSoftmaxLoss --openmetric norm
# python3 cifar10.py --loss PSoftmaxLoss --openmetric energy

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

# General MODEL parameters
parser.add_argument('--arch', default='ResNet18', choices=model_names, type=str, help='choosing network')
parser.add_argument('--embed_dim', default=128, type=int, help='embedding feature dimension')
parser.add_argument('--loss', default='SoftmaxLoss',
                    choices=["CenterLoss", "SoftmaxLoss", "ArcFaceLoss", "NormFaceLoss", "PSoftmaxLoss"],
                    type=str, help='choosing network')
parser.add_argument('--openmetric', default='possibility',
                    choices=["possibility", "distance", 'norm', 'energy', 'cosine'],
                    type=str, help='choosing network')

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
args.checkpoint = './checkpoints/cifar10/%s_%s_%s_%s_%s_dim%s-c%s-s%s-m%s' % (
    args.loss, args.openmetric, args.train_class_num, args.test_class_num, args.arch, args.embed_dim,
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
testset = CIFAR10(root='../../data', train=False, download=True, transform=transform_test,
                  train_class_num=args.train_class_num, test_class_num=args.test_class_num,
                  includes_all_train_class=args.includes_all_train_class)
# data loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4)

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
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'Train Loss', 'Train Acc.', "Test F1", 'threshold'])

    if not args.evaluate:
        for epoch in range(start_epoch, args.es):
            adjust_learning_rate(optimizer, epoch, args.lr,
                                 factor=args.lr_factor, step=args.lr_step)
            print('\nEpoch: %d | Learning rate: %f ' % (epoch + 1, optimizer.param_groups[0]['lr']))
            train_out = train(net, trainloader, optimizer, criterion, device)
            save_model(net, optimizer, epoch, os.path.join(args.checkpoint, 'last_model.pth'))
            test_out = test(net, testloader, criterion, device)
            logger.append([epoch + 1, train_out["train_loss"], train_out["accuracy"],
                           test_out["best_F1"], test_out["best_thres"]])
        logger.close()
        print(f"\nFinish training...\n")

    else:
        print("===> Evaluating ...")
        test(net, testloader, criterion, device)


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
            _, predicted = (out['cosine_fea2cen']).min(1)
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
                _, predicted = (out['cosine_fea2cen']).min(1)
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
    best_F1 = 0
    best_thres = 0
    best_eval = None

    # for these unbounded metric, we explore more intervals by *5 to achieve a relatively fair comparison.

    expand_factor = 5

    if args.openmetric == "possibility" and args.loss in ["CenterLoss", "SoftmaxLoss"]:
        threshold_min = 0.0
        threshold_max = 1.0
        openmetric_list, _ = torch.softmax(dotproduct_fea2cen_list, dim=1).max(dim=1)
        for thres in np.linspace(threshold_min, threshold_max, intervals):
            Predict_list[openmetric_list < thres] = args.train_class_num
            eval = Evaluation(Predict_list.cpu().numpy(), Target_list.cpu().numpy())
            if eval.f1_measure > best_F1:
                best_F1 = eval.f1_measure
                best_thres = thres
                best_eval = eval

    if args.openmetric == "possibility" and args.loss in ["ArcFaceLoss", "NormFaceLoss"]:
        threshold_min = 0.0
        threshold_max = 1.0
        fake_Target_list = Target_list
        fake_Target_list[fake_Target_list == args.train_class_num] = args.train_class_num - 1
        openmetric_list = criterion({"cosine_fea2cen": cosine_list}, fake_Target_list)["output"]
        openmetric_list, _ = torch.softmax(openmetric_list, dim=1).max(dim=1)
        for thres in np.linspace(threshold_min, threshold_max, intervals):
            Predict_list[openmetric_list < thres] = args.train_class_num
            eval = Evaluation(Predict_list.cpu().numpy(), Target_list.cpu().numpy())
            if eval.f1_measure > best_F1:
                best_F1 = eval.f1_measure
                best_thres = thres
                best_eval = eval

    if args.openmetric == "possibility" and args.loss in ["PSoftmaxLoss"]:
        threshold_min = 0.0
        threshold_max = 1.0
        openmetric_list, _ = torch.softmax(normweight_fea2cen_list, dim=1).max(dim=1)
        for thres in np.linspace(threshold_min, threshold_max, intervals):
            Predict_list[openmetric_list < thres] = args.train_class_num
            eval = Evaluation(Predict_list.cpu().numpy(), Target_list.cpu().numpy())
            if eval.f1_measure > best_F1:
                best_F1 = eval.f1_measure
                best_thres = thres
                best_eval = eval

    if args.openmetric == "distance" and args.loss in ["CenterLoss"]:
        openmetric_list = Distance.l2(embed_fea_list, out["centroids"])
        openmetric_list, _ = openmetric_list.min(dim=1)
        threshold_min = openmetric_list.min().item()
        threshold_max = openmetric_list.max().item()
        for thres in np.linspace(threshold_min, threshold_max, expand_factor * intervals):
            Predict_list[openmetric_list > thres] = args.train_class_num
            eval = Evaluation(Predict_list.cpu().numpy(), Target_list.cpu().numpy())
            if eval.f1_measure > best_F1:
                best_F1 = eval.f1_measure
                best_thres = thres
                best_eval = eval

    if args.openmetric == "cosine" and args.loss in ["ArcFaceLoss", "NormFaceLoss"]:
        threshold_min = 0.0
        threshold_max = 1.0
        openmetric_list, _ = cosine_list.max(dim=1)
        for thres in np.linspace(threshold_min, threshold_max, intervals):
            Predict_list[openmetric_list < thres] = args.train_class_num
            eval = Evaluation(Predict_list.cpu().numpy(), Target_list.cpu().numpy())
            if eval.f1_measure > best_F1:
                best_F1 = eval.f1_measure
                best_thres = thres
                best_eval = eval

    if args.openmetric == "norm" and args.loss in ["PSoftmaxLoss"]:

        openmetric_list = normfea_list
        threshold_min = openmetric_list.min().item()
        threshold_max = openmetric_list.max().item()
        for thres in np.linspace(threshold_min, threshold_max, expand_factor * intervals):
            Predict_list[openmetric_list < thres] = args.train_class_num
            eval = Evaluation(Predict_list.cpu().numpy(), Target_list.cpu().numpy())
            if eval.f1_measure > best_F1:
                best_F1 = eval.f1_measure
                best_thres = thres
                best_eval = eval

    if args.openmetric == "energy" and args.loss in ["PSoftmaxLoss"]:

        openmetric_list = energy_list
        threshold_min = openmetric_list.min().item()
        threshold_max = openmetric_list.max().item()
        for thres in np.linspace(threshold_min, threshold_max, expand_factor * intervals):
            Predict_list[openmetric_list < thres] = args.train_class_num
            eval = Evaluation(Predict_list.cpu().numpy(), Target_list.cpu().numpy())
            if eval.f1_measure > best_F1:
                best_F1 = eval.f1_measure
                best_thres = thres
                best_eval = eval

    print(f"Best F1 is: {best_F1}  [in best threshold: {best_thres} ]")
    return {
        "best_F1": best_F1,
        "best_thres": best_thres,
        "best_eval": best_eval
    }


if __name__ == '__main__':
    main()
