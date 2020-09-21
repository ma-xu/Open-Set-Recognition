
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

#from models import *
sys.path.append("../..")
import backbones.cifar as models
from datasets import CIFAR100
from Utils import adjust_learning_rate, progress_bar, Logger, mkdir_p, Evaluation
from CenterLoss import CenterLoss
from torch.optim import lr_scheduler
from Modelbuilder import Network

model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--arch', default='ResNet18', choices=model_names, type=str, help='choosing network')
parser.add_argument('--bs', default=256, type=int, help='batch size')
parser.add_argument('--es', default=100, type=int, help='epoch size')
parser.add_argument('--train_class_num', default=50, type=int, help='Classes used in training')
parser.add_argument('--test_class_num', default=100, type=int, help='Classes used in testing')
parser.add_argument('--includes_all_train_class', default=True,  action='store_true',
                    help='If required all known classes included in testing')
parser.add_argument('--evaluate', action='store_true',
                    help='Evaluate without training')
parser.add_argument('--centerloss_weight', default=0.003, type=float, help='center loss weight')
parser.add_argument('--center_lr', default=0.1, type=float, help='learning rate for center loss')
parser.add_argument('--threshold', default=0.1, type=float, help='threshold for center-loss probability')


args = parser.parse_args()



def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # checkpoint
    args.checkpoint = './checkpoints/cifar/' + args.arch
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=4)
    testset = CIFAR100(root='../../data', train=False, download=True, transform=transform_test,
                       train_class_num=args.train_class_num, test_class_num=args.test_class_num,
                       includes_all_train_class=args.includes_all_train_class)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4)


    # Model
    print('==> Building model..')
    net = Network(backbone=args.arch, num_classes=args.train_class_num)
    fea_dim = net.classifier.in_features
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion_softamx = nn.CrossEntropyLoss()
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
            criterion_centerloss.load_state_dict(checkpoint['centerloss'])
            # best_acc = checkpoint['acc']
            # print("BEST_ACCURACY: "+str(best_acc))
            start_epoch = checkpoint['epoch']
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'Total Loss','Softmax Loss', 'Center Loss', 'train Acc.'])

    if not args.evaluate:
        scheduler = lr_scheduler.StepLR(optimizer_softmax, step_size=30, gamma=0.1)
        for epoch in range(start_epoch, start_epoch + args.es):
            print('\nEpoch: %d   Learning rate: %f' % (epoch+1, optimizer_softmax.param_groups[0]['lr']))
            train_loss, softmax_loss, center_loss, train_acc = train(net,trainloader,optimizer_softmax,
                                     optimizer_centerloss, criterion_softamx, criterion_centerloss, device)
            save_model(net, criterion_centerloss, epoch, os.path.join(args.checkpoint,'last_model.pth'))
            logger.append([epoch+1, train_loss, softmax_loss, center_loss, train_acc])
            scheduler.step()

    test(net, testloader, device)
    logger.close()


# Training
def train(net,trainloader,optimizer_model,optimizer_centloss, criterion_softamx,criterion_centerloss,device):
    net.train()
    totoal_loss = 0
    totoal_center_loss = 0
    totoal_softmax_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        features, logits = net(inputs)
        loss_softmax = criterion_softamx(logits, targets)
        loss_center = criterion_centerloss(features, targets)
        loss = loss_softmax + args.centerloss_weight*loss_center
        optimizer_model.zero_grad()
        optimizer_centloss.zero_grad()
        loss.backward()
        optimizer_model.step()
        # XU: I dont know why.
        # by doing so, weight_cent would not impact on the learning of centers
        for param in criterion_centerloss.parameters():
            param.grad.data *= (1. / args.centerloss_weight)
        optimizer_centloss.step()

        totoal_center_loss += loss_center.item()
        totoal_softmax_loss += loss_softmax.item()
        totoal_loss += loss.item()
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader),
                     'Loss:%.3f (Softmax: %.3f CenterLoss: %.3f) Acc: %.3f%%'
            % (totoal_loss/(batch_idx+1), totoal_softmax_loss/(batch_idx+1), totoal_center_loss/(batch_idx+1),
               100.*correct/total))
    return totoal_loss/(batch_idx+1), totoal_softmax_loss/(batch_idx+1), \
           totoal_center_loss/(batch_idx+1), correct/total


def test(net, testloader, device):
    net.eval()

    scores, labels = [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            _, outputs = net(inputs)
            scores.append(outputs)
            labels.append(targets)
            progress_bar(batch_idx, len(testloader))

    # Get the prdict results.
    scores = torch.cat(scores,dim=0)
    scores = scores.softmax(dim=1)
    scores = scores.cpu().numpy()
    labels = torch.cat(labels,dim=0).cpu().numpy()

    pred = []
    for score in scores:
        pred.append(np.argmax(score) if np.max(score) >= args.threshold else args.train_class_num)

    print("Evaluation...")
    eval = Evaluation(pred, labels)
    print(f"Center-Loss accuracy is %.3f"%(eval.accuracy))

def save_model(net, centerloss, epoch, path):
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'centerloss':centerloss.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, path)

if __name__ == '__main__':
    main()

