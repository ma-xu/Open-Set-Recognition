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
from Plotter import plot_feature
from torch.optim import lr_scheduler
from Modelbuilder import Network
from CenterLoss import CenterLoss

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
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--es', default=50, type=int, help='epoch size')
parser.add_argument('--evaluate', action='store_true', help='Evaluate without training')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

# General MODEL parameters
parser.add_argument('--arch', default='LeNetPlus', choices=model_names, type=str, help='choosing network')
parser.add_argument('--embed_dim', default=2, type=int, help='embedding feature dimension')
parser.add_argument('--centerloss_weight', default=1, type=float, help='center loss weight')
parser.add_argument('--center_lr', default=0.5, type=float, help='learning rate for center loss')
parser.add_argument('--threshold', default=0.9, type=float, help='threshold for center-loss probability')




# Parameters for stage plotting
parser.add_argument('--plot', default=True, action='store_true', help='Plotting the training set.')
parser.add_argument('--plot_max', default=0, type=int, help='max examples to plot in each class, 0 indicates all.')
parser.add_argument('--plot_quality', default=200, type=int, help='DPI of plot figure')


args = parser.parse_args()


def main():
    args.checkpoint = './checkpoints/mnist/' + args.arch
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # folder to save figures
    args.plotfolder = './checkpoints/mnist/' + args.arch + '/plotter'
    if not os.path.isdir(args.plotfolder):
        mkdir_p(args.plotfolder)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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

    print('==> Building model..')
    net = Network(backbone=args.arch, num_classes=args.train_class_num,embed_dim=args.embed_dim)
    fea_dim = net.classifier.in_features
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion_softamx = nn.CrossEntropyLoss()
    print(fea_dim)
    criterion_rpl=RPLoss(number_class=args.train_class_num, feat_dim=fea_dim).to(device)

    optimizer_rpl = torch.optim.SGD(criterion_rpl.parameters(), lr=args.center_lr, momentum=0.9,
                                           weight_decay=5e-4)
    criterion_centerloss = CenterLoss(num_classes=args.train_class_num, feat_dim=fea_dim).to(device)
    optimizer_softmax = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer_centerloss = torch.optim.SGD(criterion_centerloss.parameters(), lr=args.center_lr, momentum=0.9,
                                           weight_decay=5e-4)

    if args.resume:
        # Load checkpoint.
        if os.path.isfile(args.resume):
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['net'])
            criterion_rpl.load_state_dict(checkpoint['centerloss'])
            # best_acc = checkpoint['acc']
            # print("BEST_ACCURACY: "+str(best_acc))
            start_epoch = checkpoint['epoch']
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'Total Loss', 'train Acc.'])


    if not args.evaluate:
        scheduler = lr_scheduler.StepLR(optimizer_softmax, step_size=20, gamma=0.1)
        for epoch in range(start_epoch, start_epoch + args.es):
            print('\nEpoch: %d   Learning rate: %f' % (epoch + 1, optimizer_softmax.param_groups[0]['lr']))
            train_loss, train_acc = train(net, trainloader, optimizer_softmax,
                                                                     optimizer_rpl, criterion_rpl, device)
            save_model(net, criterion_centerloss, epoch, os.path.join(args.checkpoint, 'last_model.pth'))
            # plot the training data
            # if args.plot:
            #     plot_feature(net,criterion_centerloss, trainloader, device, args.plotfolder, epoch=epoch,
            #              plot_class_num=args.train_class_num,maximum=args.plot_max, plot_quality=args.plot_quality)

            logger.append([epoch + 1, train_loss,  train_acc])
            scheduler.step()
            test(net, testloader,criterion_rpl, device)



    # if args.plot:
    #     plot_feature(net, criterion_centerloss, testloader, device, args.plotfolder, epoch="test",
    #                  plot_class_num=args.train_class_num+1, maximum=args.plot_max, plot_quality=args.plot_quality)
    logger.close()


# Training
def train(net, trainloader, optimizer_model, optimizer_rpl, criterion_rpl, device):
    net.train()
    totoal_loss = 0
    totoal_center_loss = 0
    totoal_softmax_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        features, logits = net(inputs)
        _, loss = criterion_rpl(features, targets)
        optimizer_model.zero_grad()
        optimizer_rpl.zero_grad()
        loss.backward()
        optimizer_model.step()
        # XU: I dont know why.
        # by doing so, weight_cent would not impact on the learning of centers
        # for param in criterion_rpl.parameters():
        #     param.grad.data *= (1. / args.centerloss_weight)
        optimizer_rpl.step()

        totoal_loss += loss.item()

        logits = torch.matmul(features, torch.transpose(criterion_rpl.Dist.centers, 1, 0))


        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader),
                     'Loss:%.3f  Acc: %.3f%%'
                     % (totoal_loss / (batch_idx + 1),
                        100. * correct / total))
    return totoal_loss / (batch_idx + 1), correct / total


def test(net, testloader,criterion_rpl, device):
    net.eval()

    scores, labels ,logits = [], [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            features, outputs = net(inputs)
            _logits, _ = criterion_rpl(features)
            scores.append(outputs)
            labels.append(targets)
            logits.append(_logits)
            progress_bar(batch_idx, len(testloader))

    # Get the prdict results.
    scores = torch.cat(scores, dim=0)
    print(f"scores.shape: {scores.shape}")
    scores = scores.softmax(dim=1)
    scores = scores.cpu().numpy()
    logits = torch.cat(logits, dim=0)

    print(f"logits.shape: {logits.shape}")
    logits = logits.cpu.numpy()
    scores = logits
    labels = torch.cat(labels, dim=0).cpu().numpy()

    pred = []
    for score in scores:
        pred.append(np.argmax(score) if np.max(score) >= args.threshold else args.train_class_num)

    print("Evaluation...")
    eval = Evaluation(pred, labels, scores)
    torch.save(eval, os.path.join(args.checkpoint, 'eval.pkl'))
    print(f"Center-Loss accuracy is %.3f" % (eval.accuracy))
    print(f"Center-Loss F1 is %.3f" % (eval.f1_measure))
    print(f"Center-Loss f1_macro is %.3f" % (eval.f1_macro))
    print(f"Center-Loss f1_macro_weighted is %.3f" % (eval.f1_macro_weighted))
    print(f"Center-Loss area_under_roc is %.3f" % (eval.area_under_roc))

def save_model(net, centerloss, epoch, path):
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'centerloss': centerloss.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, path)



class RPLoss(nn.CrossEntropyLoss):
    def __init__(self, number_class, feat_dim):
        super(RPLoss, self).__init__()
        self.weight_pl = 0.1
        self.temp = 1
        self.Dist = Dist(num_classes=number_class, feat_dim=feat_dim, num_centers=1)
        self.radius = 1

        self.radius = nn.Parameter(torch.Tensor(self.radius))
        self.radius.data.fill_(0)

    def forward(self, x, labels=None):
        dist = self.Dist(x)
        logits = F.softmax(dist, dim=1)
        if labels is None: return logits, 0
        loss = F.cross_entropy(dist / self.temp, labels)
        center_batch = self.Dist.centers[labels, :]
        _dis = (x - center_batch).pow(2).mean(1)
        loss_r = F.mse_loss(_dis, self.radius)
        loss = loss + self.weight_pl * loss_r

        return logits, loss


class Dist(nn.Module):
    def __init__(self, num_classes=10, num_centers=1, feat_dim=2, init='random'):
        super(Dist, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_centers = num_centers

        if init == 'random':
            self.centers = nn.Parameter(0.1 * torch.randn(num_classes * num_centers, self.feat_dim))
        else:
            self.centers = nn.Parameter(torch.Tensor(num_classes * num_centers, self.feat_dim))
            self.centers.data.fill_(0)

    def forward(self, features, center=None, metric='l2'):

        if metric == 'l2':
            f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
            if center is None:
                c_2 = torch.sum(torch.pow(self.centers, 2), dim=1, keepdim=True)
                # print(c_2.shape)
                # print(f_2.shape)
                # print(torch.transpose(self.centers, 1, 0).shape)
                # print(features.shape)
                # print(torch.matmul(features, torch.transpose(self.centers, 1, 0)).shape)


                dist = f_2 - 2*torch.matmul(features, torch.transpose(self.centers, 1, 0)) + torch.transpose(c_2, 1, 0)
            else:
                c_2 = torch.sum(torch.pow(center, 2), dim=1, keepdim=True)
                dist = f_2 - 2*torch.matmul(features, torch.transpose(center, 1, 0)) + torch.transpose(c_2, 1, 0)
            dist = dist / float(features.shape[1])
        else:
            if center is None:
                center = self.centers
            else:
                center = center
            dist = features.matmul(center.t())
        dist = torch.reshape(dist, [-1, self.num_classes, self.num_centers])
        dist = torch.mean(dist, dim=2)

        return dist




if __name__ == '__main__':
    main()
