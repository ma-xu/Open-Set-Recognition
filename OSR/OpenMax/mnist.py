
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
from datasets import CIFAR100,MNIST
from Utils import adjust_learning_rate, progress_bar, Logger, mkdir_p, Evaluation
from openmax import compute_train_score_and_mavs_and_dists,fit_weibull,openmax
from Modelbuilder import Network
from Plotter import plot_feature

model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--arch', default='LeNetPlus', choices=model_names, type=str, help='choosing network')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--es', default=50, type=int, help='epoch size')
parser.add_argument('--train_class_num', default=7, type=int, help='Classes used in training')
parser.add_argument('--test_class_num', default=10, type=int, help='Classes used in testing')
parser.add_argument('--includes_all_train_class', default=True,  action='store_true',
                    help='If required all known classes included in testing')
parser.add_argument('--embed_dim', default=2, type=int, help='embedding feature dimension')
parser.add_argument('--evaluate', action='store_true',
                    help='Evaluate without training')

#Parameters for weibull distribution fitting.
parser.add_argument('--weibull_tail', default=20, type=int, help='Classes used in testing')
parser.add_argument('--weibull_alpha', default=3, type=int, help='Classes used in testing')
parser.add_argument('--weibull_threshold', default=0.9, type=float, help='Classes used in testing')


# Parameters for stage plotting
parser.add_argument('--plot', default=True, action='store_true', help='Plotting the training set.')
parser.add_argument('--plot_max', default=0, type=int, help='max examples to plot in each class, 0 indicates all.')
parser.add_argument('--plot_quality', default=200, type=int, help='DPI of plot figure')


args = parser.parse_args()



def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # checkpoint
    args.checkpoint = './checkpoints/mnist/' + args.arch
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # folder to save figures
    args.plotfolder = './checkpoints/mnist/' + args.arch + '/plotter'
    if not os.path.isdir(args.plotfolder):
        mkdir_p(args.plotfolder)

    # Data
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


    # Model
    net = Network(backbone=args.arch, num_classes=args.train_class_num, embed_dim=args.embed_dim)
    fea_dim = net.classifier.in_features
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        if os.path.isfile(args.resume):
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['net'])
            # best_acc = checkpoint['acc']
            # print("BEST_ACCURACY: "+str(best_acc))
            start_epoch = checkpoint['epoch']
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'Learning Rate', 'Train Loss','Train Acc.', 'Test Loss', 'Test Acc.'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # test(0, net, trainloader, testloader, criterion, device)
    epoch=0
    if not args.evaluate:
        for epoch in range(start_epoch, start_epoch + args.es):
            print('\nEpoch: %d   Learning rate: %f' % (epoch+1, optimizer.param_groups[0]['lr']))
            adjust_learning_rate(optimizer, epoch, args.lr, step=20)
            train_loss, train_acc = train(net,trainloader,optimizer,criterion,device)
            save_model(net, None, epoch, os.path.join(args.checkpoint,'last_model.pth'))
            test_loss, test_acc = 0, 0
            #
            logger.append([epoch+1, optimizer.param_groups[0]['lr'], train_loss, train_acc, test_loss, test_acc])
            plot_feature(net, trainloader, device, args.plotfolder, epoch=epoch,
                         plot_class_num=args.train_class_num, maximum=args.plot_max, plot_quality=args.plot_quality)

    test(epoch, net, trainloader, testloader, criterion, device)
    plot_feature(net, testloader, device, args.plotfolder, epoch="test",
                 plot_class_num=args.train_class_num+1, maximum=args.plot_max, plot_quality=args.plot_quality)
    logger.close()


# Training
def train(net,trainloader,optimizer,criterion,device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        _, outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1), correct/total


def test(epoch, net,trainloader,  testloader,criterion, device):
    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    scores, labels = [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            # loss = criterion(outputs, targets)
            # test_loss += loss.item()
            # _, predicted = outputs.max(1)
            scores.append(outputs)
            labels.append(targets)

            # total += targets.size(0)
            # correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader))

    # Get the prdict results.
    scores = torch.cat(scores,dim=0).cpu().numpy()
    labels = torch.cat(labels,dim=0).cpu().numpy()
    scores = np.array(scores)[:, np.newaxis, :]
    labels = np.array(labels)

    # Fit the weibull distribution from training data.
    print("Fittting Weibull distribution...")
    _, mavs, dists = compute_train_score_and_mavs_and_dists(args.train_class_num, trainloader, device, net)
    categories = list(range(0, args.train_class_num))
    weibull_model = fit_weibull(mavs, dists, categories, args.weibull_tail, "euclidean")

    pred_softmax, pred_softmax_threshold, pred_openmax = [], [], []
    for score in scores:
        so, ss = openmax(weibull_model, categories, score,
                         0.5, args.weibull_alpha, "euclidean")  # openmax_prob, softmax_prob
        pred_softmax.append(np.argmax(ss))
        pred_softmax_threshold.append(np.argmax(ss) if np.max(ss) >= args.weibull_threshold else args.train_class_num)
        pred_openmax.append(np.argmax(so) if np.max(so) >= args.weibull_threshold else args.train_class_num)

    print("Evaluation...")
    eval_softmax = Evaluation(pred_softmax, labels)
    eval_softmax_threshold = Evaluation(pred_softmax_threshold, labels)
    eval_openmax = Evaluation(pred_openmax, labels)



    print(f"Softmax accuracy is %.3f"%(eval_softmax.accuracy))
    print(f"Softmax-with-threshold accuracy is %.3f"%(eval_softmax_threshold.accuracy))
    print(f"Openmax accuracy is %.3f"%(eval_openmax.accuracy))

def save_model(net, acc, epoch, path):
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'testacc': acc,
        'epoch': epoch,
    }
    torch.save(state, path)

if __name__ == '__main__':
    main()

