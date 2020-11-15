from __future__ import print_function

import torch
import torch.nn as nn
import math
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
from DFPLoss import DFPLoss, DFPLoss2
from DFPNet import DFPNet
from MyPlotter import plot_feature, plot_distance,plot_gap
from helper import get_gap_stat

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
parser.add_argument('--embed_dim', default=512, type=int, help='embedding feature dimension')
parser.add_argument('--distance', default='l2', choices=['l2', 'l1', 'cosine', 'dotproduct'],
                    type=str, help='choosing distance metric')
parser.add_argument('--similarity', default='dotproduct', choices=['l2', 'l1', 'cosine', 'dotproduct'],
                    type=str, help='choosing distance metric')
parser.add_argument('--alpha', default=1.0, type=float, help='weight of distance loss')
parser.add_argument('--beta', default=1.0, type=float, help='weight of center between  loss')
parser.add_argument('--theta', default=10.0, type=float, help='slope for input data distance within/out thresholds,'
                                                             'default 10.')

parser.add_argument('--scaled', default='True',  action='store_true',
                    help='If scale distance by sqrt(embed_dim)')
parser.add_argument('--norm_centroid', action='store_true', help='Normalize the centroid using L2-normailization')
parser.add_argument('--decorrelation', action='store_true', help='Normalize the centroid using L2-normailization')


# Parameters for stage 1
parser.add_argument('--stage1_resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--stage1_es', default=50, type=int, help='epoch size')
parser.add_argument('--stage1_lr', default=0.1, type=float, help='learning rate')  # works for MNIST

# Parameters for stage 2
parser.add_argument('--stage2_resume',default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--stage2_es', default=50, type=int, help='epoch size')
parser.add_argument('--stage2_lr', default=0.01, type=float, help='learning rate')  # works for MNIST

# Parameters for stage plotting
parser.add_argument('--plot', action='store_true', help='Plotting the training set.')
parser.add_argument('--plot_max', default=0, type=int, help='max examples to plot in each class, 0 indicates all.')

parser.add_argument('--plot_quality', default=200, type=int, help='DPI of plot figure')
parser.add_argument('--bins', default=50, type=int, help='divided into n bins')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.checkpoint = './checkpoints/cifar/' + \
                  '/%s-%s_%s_%s-%s-%s_%s_%s' % (args.train_class_num, args.test_class_num, args.arch, args.alpha,
                                             args.beta, args.theta, args.embed_dim, str(args.decorrelation))
if not os.path.isdir(args.checkpoint):
    mkdir_p(args.checkpoint)

# folder to save figures
args.plotfolder1 = os.path.join(args.checkpoint, "plotter_Stage1")
if not os.path.isdir(args.plotfolder1):
    mkdir_p(args.plotfolder1)
# folder to save figures
args.plotfolder2 = os.path.join(args.checkpoint, "plotter_Stage2")
if not os.path.isdir(args.plotfolder2):
    mkdir_p(args.plotfolder2)

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4)


def main():
    print(device)
    stage1_dict = {
        'distance': {'thresholds': torch.ones(args.train_class_num)},
        'stat': None,
        'net': None
    }

    if not args.evaluate and not os.path.isfile(args.stage2_resume):
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
                 distance=args.distance, similarity=args.similarity, scaled=args.scaled,
                 norm_centroid=args.norm_centroid, decorrelation=args.decorrelation)

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
        logger.set_names(['Epoch', 'Train Loss', 'Similarity Loss', 'Distance Loss', 'Train Acc.'])

    # after resume
    criterion = DFPLoss(alpha=args.alpha)
    optimizer = optim.SGD(net.parameters(), lr=args.stage1_lr, momentum=0.9, weight_decay=5e-4)


    for epoch in range(start_epoch, args.stage1_es):
        adjust_learning_rate(optimizer, epoch, args.stage1_lr, step=20)
        print('\nStage_1 Epoch: %d | Learning rate: %f ' % (epoch + 1, optimizer.param_groups[0]['lr']))
        train_out = stage1_train(net, trainloader, optimizer, criterion, device)
        save_model(net, epoch, os.path.join(args.checkpoint, 'stage_1_last_model.pth'))
        logger.append([epoch + 1, train_out["train_loss"], train_out["loss_similarity"],
                       train_out["loss_distance"], train_out["accuracy"]])

    # calculating distances for last epoch
    distance_results = plot_distance(net, trainloader, device, args)
    # print(f"the distance thresholds are\n {distance_results['thresholds']}\n")
    # gap_results = plot_gap(net, trainloader, device, args)
    # stat = get_gap_stat(net, trainloader, device, args)
    # estimator =CGD_estimator(gap_results)

    logger.close()
    print(f"\nFinish Stage-1 training...\n")
    print("===> Evaluating ...")
    stage1_test(net, testloader, device)

    return {"net": net,
            "distance": distance_results,
            # "stat": stat
            }


# Training
def stage1_train(net, trainloader, optimizer, criterion, device):
    net.train()
    train_loss = 0
    loss_similarity = 0
    loss_distance = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        out = net(inputs)
        loss_dict = criterion(out, targets)
        loss = loss_dict['total']
        # loss = loss_dict['similarity']
        # loss = loss_dict['distance']
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loss_similarity += (loss_dict['similarity']).item()
        loss_distance += (loss_dict['distance']).item()

        _, predicted = (out['sim_fea2cen']).max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return {
        "train_loss": train_loss / (batch_idx + 1),
        "loss_similarity": loss_similarity / (batch_idx + 1),
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


def main_stage2(stage1_dict):
    print('==> Building stage2 model..')
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    net = DFPNet(backbone=args.arch, num_classes=args.train_class_num, embed_dim=args.embed_dim,
                 distance=args.distance, similarity=args.similarity, scaled=args.scaled,
                 norm_centroid=args.norm_centroid, decorrelation=args.decorrelation)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if not args.evaluate and not os.path.isfile(args.stage2_resume):
        net = stage1_dict['net']
        net = net.to(device)
        thresholds = stage1_dict['distance']['thresholds']
        # stat = stage1_dict["stat"]
        net.module.set_threshold(thresholds.to(device))


    if args.stage2_resume:
        # Load checkpoint.
        if os.path.isfile(args.stage2_resume):
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(args.stage2_resume)
            net.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch']
            try:
                thresholds = checkpoint['net']['thresholds']
            except:
                thresholds = checkpoint['net']['module.thresholds']
            net.module.set_threshold(thresholds.to(device))


            logger = Logger(os.path.join(args.checkpoint, 'log_stage2.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log_stage2.txt'))
        logger.set_names(['Epoch', 'Train Loss', 'Similarity Loss', 'Distance in', 'Distance out',
                          'Distance Center', 'Train Acc.'])



    if args.evaluate:
        stage2_test(net, testloader, device)
        return net

    # after resume
    criterion = DFPLoss2(alpha=args.alpha,beta=args.beta, theta=args.theta)
    optimizer = optim.SGD(net.parameters(), lr=args.stage1_lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(start_epoch, args.stage2_es):
        print('\nStage_2 Epoch: %d   Learning rate: %f' % (epoch + 1, optimizer.param_groups[0]['lr']))
        # Here, I didn't set optimizers respectively, just for simplicity. Performance did not vary a lot.
        adjust_learning_rate(optimizer, epoch, args.stage2_lr, step=20)
        # if epoch %5 ==0:
        #     distance_results = plot_distance(net, trainloader, device, args)
        #     thresholds = distance_results['thresholds']
        #     net.module.set_threshold(thresholds.to(device))
        train_out = stage2_train(net, trainloader, optimizer, criterion, device)
        save_model(net, epoch, os.path.join(args.checkpoint, 'stage_2_last_model.pth'))
        stage2_test(net, testloader, device)
        # stat = get_gap_stat(net2, trainloader, device, args)

        logger.append([epoch + 1, train_out["train_loss"], train_out["loss_similarity"],
                       train_out["distance_in"], train_out["distance_out"],
                       train_out["distance_center"], train_out["accuracy"]])

    print(f"\nFinish Stage-2 training...\n")

    logger.close()
    stage2_test(net, testloader, device)
    return net


def stage2_train(net2, trainloader, optimizer, criterion, device):
    net2.train()
    train_loss = 0
    loss_similarity = 0
    distance_in = 0
    distance_out = 0
    distance_center = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        out = net2(inputs)
        loss_dict = criterion(out, targets)
        loss = loss_dict['total']
        # loss = loss_dict['similarity']
        # loss = loss_dict['distance']
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loss_similarity += (loss_dict['similarity']).item()
        distance_in += (loss_dict['distance_in']).item()
        distance_out += (loss_dict['distance_out']).item()
        distance_center += (loss_dict['distance_center']).item()

        _, predicted = (out['sim_fea2cen']).max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return {
        "train_loss": train_loss / (batch_idx + 1),
        "loss_similarity": loss_similarity / (batch_idx + 1),
        "distance_in": distance_in / (batch_idx + 1),
        "distance_out": distance_out / (batch_idx + 1),
        "distance_center": distance_center / (batch_idx + 1),
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


def stage2_test(net, testloader, device):
    correct = 0
    total = 0

    pred_list, target_list, score_list = [], [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            out = net(inputs)
            threshold = out["thresholds"]  # [class]
            sim_fea2cen= out["sim_fea2cen"]  # [batch,class]
            dis_fea2cen= out["dis_fea2cen"]  # [batch,class]

            b,c = dis_fea2cen.shape
            dis_predicted, predicted_dis_ind = (dis_fea2cen).min(1) #[b]
            sim_predicted, predicted =  (sim_fea2cen).max(1) #[b]


            compare_threshold = 1*threshold[predicted]
            ind1 = (dis_predicted-compare_threshold)>0
            ind2 = (1/math.sqrt(c) - sim_predicted) >0
            print(f"ind1 type {type(ind1)} value {ind1}")
            print(f"ind2 type {type(ind2)} value {ind2}")

            # predicted[unkown_ind] = c

            pred_list.extend(predicted.tolist())
            target_list.extend(targets.tolist())
            score_list.extend(sim_fea2cen.tolist())

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()



            progress_bar(batch_idx, len(testloader), '| Acc: %.3f%% (%d/%d)'
                         % (100. * correct / total, correct, total))
    eval_result = Evaluation(pred_list, target_list, score_list)

    torch.save(eval_result, os.path.join(args.checkpoint, 'eval.pkl'))

    print(f"accuracy is %.3f" % (eval_result.accuracy))
    print(f"F1 is %.3f" % (eval_result.f1_measure))
    print(f"f1_macro is %.3f" % (eval_result.f1_macro))
    print(f"f1_macro_weighted is %.3f" % (eval_result.f1_macro_weighted))
    print(f"area_under_roc is %.3f" % (eval_result.area_under_roc))
    print(f"confuse matrix unkown is {eval_result.confusion_matrix[:,-1]}")



if __name__ == '__main__':
    main()
