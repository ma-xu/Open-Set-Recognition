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
from mnist_plotter import plot_feature

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
parser.add_argument('--beta', default=1.0, type=float, help='wight of between-class distance loss')
parser.add_argument('--alpha', default=1.0, type=float, help='weight of total distance loss')
parser.add_argument('--distance', default='l2', choices=['l2', 'l1', 'dotproduct'],
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



args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.checkpoint = './checkpoints/mnist/' + args.arch
if not os.path.isdir(args.checkpoint):
    mkdir_p(args.checkpoint)

# folder to save figures
args.plotfolder = './checkpoints/mnist/' + args.arch + '/plotter_%s_%s' % (args.alpha, args.beta)
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
    criterion_cls = nn.CrossEntropyLoss()
    criterion_dis = DFPLoss(beta=args.beta)
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
        logger = Logger(os.path.join(args.checkpoint, 'log_stage1_%s_%s.txt' % (args.alpha, args.beta)))
        logger.set_names(['Epoch', 'Train Loss', 'Softmax Loss', 'Distance Loss',
                          'Within Loss', 'Between Loss', 'Train Acc.'])

    for epoch in range(start_epoch, start_epoch + args.stage1_es):
        print('\nStage_1 Epoch: %d | Learning rate: %f ' % (epoch + 1, optimizer.param_groups[0]['lr']))
        adjust_learning_rate(optimizer, epoch, args.stage1_lr, step=30)
        train_out = stage1_train(
            net, trainloader, optimizer, criterion_cls, criterion_dis, device)
        save_model(net, epoch, os.path.join(args.checkpoint,
                                                           'stage_1_last_model_%s_%s.pth' % (args.alpha, args.beta)))
        # ['Epoch', 'Train Loss', 'Softmax Loss', 'Distance Loss', 'Within Loss', 'Between Loss', 'Train Acc.']
        logger.append([epoch + 1, train_out["train_loss"], train_out["cls_loss"],
                       train_out["dis_loss_total"], train_out["dis_loss_within"],
                       train_out["dis_loss_between"], train_out["accuracy"]])
        if args.plot:
            plot_feature(net, trainloader, device, args.plotfolder, epoch=epoch,
                         plot_class_num=args.train_class_num, maximum=args.plot_max)
    logger.close()
    print(f"\nFinish Stage-1 training...\n")
    return net


# Training
def stage1_train(net, trainloader, optimizer, criterion_cls, criterion_dis, device):
    net.train()
    train_loss = 0
    cls_loss = 0
    dis_loss_total = 0
    dis_loss_within = 0
    dis_loss_between = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        out = net(inputs)
        loss_cls = criterion_cls(out["logits"], targets)
        # loss_dis = loss_dis_within + loss_dis_between
        loss_dis = criterion_dis(out["dist_fea2cen"], targets)
        loss = loss_cls + args.alpha * (loss_dis["total"])
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        cls_loss += loss_cls.item()
        dis_loss_total += loss_dis["total"].item()
        dis_loss_within += loss_dis["within"].item()
        dis_loss_between += loss_dis["between"].item()

        _, predicted = (out["logits"]).max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return {
        "train_loss": train_loss / (batch_idx + 1),
        "cls_loss": cls_loss / (batch_idx + 1),
        "dis_loss_total": dis_loss_total / (batch_idx + 1),
        "dis_loss_within": dis_loss_within / (batch_idx + 1),
        "dis_loss_between": dis_loss_between / (batch_idx + 1),
        "accuracy": correct / total
    }


def stage2_train(net, trainloader, optimizer, criterion, fea_criterion, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, _, _, features = net(inputs)
        loss = criterion(outputs, targets)
        loss_fea = fea_criterion(features, targets)
        loss += loss_fea * args.stage2_fea_loss_weight
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return train_loss / (batch_idx + 1), correct / total


# calculate centroids
def cal_centroids(net, device):
    print(f"===> Calculating centroids ...")
    # data loader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=4)

    net.eval()
    centroids = torch.zeros([args.train_class_num, args.stage1_feature_dim]).to(device)
    class_count = torch.zeros([args.train_class_num, 1]).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # outputs, _, _ = net(inputs)
            _, features, _, _ = net(inputs)
            for i in range(0, targets.size(0)):
                label = targets[i]
                class_count[label] += 1
                centroids[label] += features[i, :]
    centroids = centroids / (class_count.expand_as(centroids))
    return centroids


#
# def main_stage2(net1, centroids):
#
#     print(f"\n===> Start Stage-2 training...\n")
#     start_epoch = 0  # start from epoch 0 or last checkpoint epoch
#     # Ignore the classAwareSampler since we are not focusing on long-tailed problem.
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True,  num_workers=4)
#     print('==> Building model..')
#     net2 = Network(backbone=args.arch, embed_dim=512, num_classes=args.train_class_num,
#                   use_fc=True, attmodule=True, classifier='metaembedding', backbone_fc=False, data_shape=4)
#     net2 = net2.to(device)
#     if not args.evaluate:
#         init_stage2_model(net1, net2)
#
#     criterion = nn.CrossEntropyLoss()
#     fea_criterion = DiscCentroidsLoss(args.train_class_num, args.stage1_feature_dim)
#     fea_criterion = fea_criterion.to(device)
#     optimizer = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
#
#     # passing centroids data.
#     if not args.evaluate:
#         pass_centroids(net2, fea_criterion, init_centroids=centroids)
#
#     if device == 'cuda':
#         net2 = torch.nn.DataParallel(net2)
#         cudnn.benchmark = True
#
#     if args.stage2_resume:
#         # Load checkpoint.
#         if os.path.isfile(args.stage2_resume):
#             print('==> Resuming from checkpoint..')
#             checkpoint = torch.load(args.stage2_resume)
#             net2.load_state_dict(checkpoint['net'])
#             # best_acc = checkpoint['acc']
#             # print("BEST_ACCURACY: "+str(best_acc))
#             start_epoch = checkpoint['epoch']
#             logger = Logger(os.path.join(args.checkpoint, 'log_stage2.txt'), resume=True)
#         else:
#             print("=> no checkpoint found at '{}'".format(args.resume))
#     else:
#         logger = Logger(os.path.join(args.checkpoint, 'log_stage2.txt'))
#         logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Train Acc.'])
#
#     if not args.evaluate:
#         for epoch in range(start_epoch, start_epoch + args.stage2_es):
#             print('\nStage_2 Epoch: %d   Learning rate: %f' % (epoch + 1, optimizer.param_groups[0]['lr']))
#             # Here, I didn't set optimizers respectively, just for simplicity. Performance did not vary a lot.
#             adjust_learning_rate(optimizer, epoch, args.lr, step=20)
#             train_loss, train_acc = stage2_train(net2, trainloader, optimizer, criterion, fea_criterion, device)
#             save_model(net2, None, epoch, os.path.join(args.checkpoint, 'stage_2_last_model.pth'))
#             logger.append([epoch + 1, optimizer.param_groups[0]['lr'], train_loss, train_acc])
#             pass_centroids(net2, fea_criterion, init_centroids=None)
#         print(f"\nFinish Stage-2 training...\n")
#     logger.close()
#
#
#
#     testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4)
#     test(net2, testloader, device)
#     return net2
#


def init_stage2_model(net1, net2):
    # net1: net from stage 1.
    # net2: net from stage 2.
    dict1 = net1.state_dict()
    dict2 = net2.state_dict()
    for k, v in dict1.items():
        if k.startswith("module.1."):
            k = k[9:]  # remove module.1.
        if k.startswith("module."):
            k = k[7:]  # remove module.1.
        if k.startswith("classifier"):
            continue  # we do not load the classifier weight from stage 1.
        dict2[k] = v
    net2.load_state_dict(dict2)


def pass_centroids(net2, fea_criterion, init_centroids=None):
    # net2: model in stage 2
    # fea_criterion: the centroidsLoss
    # init_centroids: initiated centroids from stage1(training set)
    if init_centroids is not None:
        centroids = init_centroids
        criterion_dict = fea_criterion.state_dict()
        criterion_dict['centroids'] = centroids
        fea_criterion.load_state_dict(criterion_dict)
    else:
        criterion_dict = fea_criterion.state_dict()
        centroids = criterion_dict['centroids']
    net2_dict = net2.state_dict()
    # in case module or module.1.
    for k, _ in net2_dict.items():
        if k.__contains__('classifier.centroids'):
            net2_dict[k] = centroids

    net2.load_state_dict(net2_dict)


def test(net, testloader, device):
    net.eval()
    scores, labels = [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _, _, _ = net(inputs)
            scores.append(outputs)
            labels.append(targets)
            progress_bar(batch_idx, len(testloader))

    scores = torch.cat(scores, dim=0)
    scores = scores.softmax(dim=1)
    scores = scores.cpu().numpy()

    print(scores.shape)
    labels = torch.cat(labels, dim=0).cpu().numpy()
    pred = []
    for score in scores:
        pred.append(np.argmax(score) if np.max(score) >= args.oltr_threshold else args.train_class_num)
    eval = Evaluation(pred, labels)
    print(f"OLTR accuracy is %.3f" % (eval.accuracy))


def save_model(net, epoch, path):
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, path)


if __name__ == '__main__':
    main()
