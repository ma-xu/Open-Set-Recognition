
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
from netbuilder import Network
from DiscCentroidsLoss import DiscCentroidsLoss

model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

# Dataset preperation
parser.add_argument('--train_class_num', default=50, type=int, help='Classes used in training')
parser.add_argument('--test_class_num', default=100, type=int, help='Classes used in testing')
parser.add_argument('--includes_all_train_class', default=True,  action='store_true',
                    help='If required all known classes included in testing')

# Others
parser.add_argument('--arch', default='ResNet18', choices=model_names, type=str, help='choosing network')
parser.add_argument('--bs', default=256, type=int, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--evaluate', action='store_true', help='Evaluate without training')

# Parameters for stage 1
parser.add_argument('--stage1_resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--stage1_es', default=35, type=int, help='epoch size')
parser.add_argument('--stage1_use_fc', default=False,  action='store_true',
                    help='If to use the last FC/embedding layer in network, FC (whatever, stage1_feature_dim)')
parser.add_argument('--stage1_feature_dim', default=512, type=int, help='embedding feature dimension')
parser.add_argument('--stage1_classifier', default='dotproduct', type=str,choices=['dotproduct', 'cosnorm', 'metaembedding'],
                    help='Select a classifier (default dotproduct)')


# Parameters for stage 2
parser.add_argument('--stage2_resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--stage2_es', default=70, type=int, help='epoch size')
parser.add_argument('--stage2_use_fc', default=True,  action='store_true',
                    help='If to use the last FC/embedding layer in network, FC (whatever, stage1_feature_dim)')
parser.add_argument('--stage2_fea_loss_weight', default=0.01, type=float, help='The wegiht for feature loss')
parser.add_argument('--oltr_threshold', default=0.9, type=float, help='The score threshold for OLTR')



args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.checkpoint = './checkpoints/cifar/' + args.arch
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

trainset = CIFAR100(root='../../data', train=True, download=True, transform=transform_train,
                    train_class_num=args.train_class_num, test_class_num=args.test_class_num,
                    includes_all_train_class=args.includes_all_train_class)

testset = CIFAR100(root='../../data', train=False, download=True, transform=transform_test,
                   train_class_num=args.train_class_num, test_class_num=args.test_class_num,
                   includes_all_train_class=args.includes_all_train_class)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4)

# ensure load checkpoints for evaluation
if args.evaluate:
    assert os.path.isfile(args.stage2_resume)


def main():
    print(device)
    net1,centroids = None,None
    if not args.evaluate:
        net1 = main_stage1()
        centroids = cal_centroids(net1, device)
    main_stage2(net1, centroids)


def main_stage1():
    print(f"\nStart Stage-1 training...\n")
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    # data loader


    # Model
    print('==> Building model..')
    net = Network(backbone=args.arch, embed_dim=512, num_classes=args.train_class_num,
                 use_fc=False, attmodule=False, classifier='dotproduct', backbone_fc=False, data_shape=4)
    # net = models.__dict__[args.arch](num_classes=args.train_class_num) # CIFAR 100
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
            # best_acc = checkpoint['acc']
            # print("BEST_ACCURACY: "+str(best_acc))
            start_epoch = checkpoint['epoch']
            logger = Logger(os.path.join(args.checkpoint, 'log_stage1.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log_stage1.txt'))
        logger.set_names(['Epoch', 'Learning Rate', 'Train Loss','Train Acc.'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(start_epoch, args.stage1_es):
        print('\nStage_1 Epoch: %d   Learning rate: %f' % (epoch+1, optimizer.param_groups[0]['lr']))
        adjust_learning_rate(optimizer, epoch, args.lr,step=10)
        train_loss, train_acc = stage1_train(net,trainloader,optimizer,criterion,device)
        save_model(net, None, epoch, os.path.join(args.checkpoint,'stage_1_last_model.pth'))
        logger.append([epoch+1, optimizer.param_groups[0]['lr'], train_loss, train_acc])
    logger.close()
    print(f"\nFinish Stage-1 training...\n")
    return net

# Training
def stage1_train(net,trainloader,optimizer,criterion,device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, _, _, _ = net(inputs)
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


def stage2_train(net,trainloader,optimizer,optimizer2,
                 criterion, fea_criterion, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        optimizer2.zero_grad()
        outputs, _, _, features = net(inputs)
        loss = criterion(outputs, targets)
        loss_fea = fea_criterion(features, targets)
        loss += loss_fea*args.stage2_fea_loss_weight
        loss.backward()
        optimizer.step()
        optimizer2.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1), correct/total


# calculate centroids
def cal_centroids(net,device):
    print(f"===> Calculating centroids ...")
    # data loader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=4)

    net.eval()
    centroids = torch.zeros([args.train_class_num,args.stage1_feature_dim]).to(device)
    class_count = torch.zeros([args.train_class_num,1]).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # outputs, _, _ = net(inputs)
            _, features, _,_ = net(inputs)
            for i in range(0,targets.size(0)):
                label = targets[i]
                class_count[label] += 1
                centroids[label] += features[i, :]
    centroids = centroids/(class_count.expand_as(centroids))
    return centroids


def main_stage2(net1, centroids):

    print(f"\n===> Start Stage-2 training...\n")
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    # Ignore the classAwareSampler since we are not focusing on long-tailed problem.
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True,  num_workers=4)
    print('==> Building model..')
    net2 = Network(backbone=args.arch, embed_dim=512, num_classes=args.train_class_num,
                  use_fc=True, attmodule=True, classifier='metaembedding', backbone_fc=False, data_shape=4)
    net2 = net2.to(device)
    if not args.evaluate:
        init_stage2_model(net1, net2)

    criterion = nn.CrossEntropyLoss()
    fea_criterion = DiscCentroidsLoss(args.train_class_num, args.stage1_feature_dim)
    fea_criterion = fea_criterion.to(device)
    optimizer = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer_criterion = optim.SGD(fea_criterion.parameters(), lr=args.lr * 0.1, momentum=0.9, weight_decay=5e-4)

    # passing centroids data.
    if not args.evaluate:
        pass_centroids(net2, fea_criterion, init_centroids=centroids)

    if device == 'cuda':
        net2 = torch.nn.DataParallel(net2)
        cudnn.benchmark = True

    if args.stage2_resume:
        # Load checkpoint.
        if os.path.isfile(args.stage2_resume):
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(args.stage2_resume)
            net2.load_state_dict(checkpoint['net'])
            # best_acc = checkpoint['acc']
            # print("BEST_ACCURACY: "+str(best_acc))
            start_epoch = checkpoint['epoch']
            logger = Logger(os.path.join(args.checkpoint, 'log_stage2.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log_stage2.txt'))
        logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Train Acc.'])

    if not args.evaluate:
        for epoch in range(start_epoch, args.stage2_es):
            print('\nStage_2 Epoch: %d   Learning rate: %f' % (epoch + 1, optimizer.param_groups[0]['lr']))
            # Here, I didn't set optimizers respectively, just for simplicity. Performance did not vary a lot.
            adjust_learning_rate(optimizer, epoch, args.lr, step=20)
            train_loss, train_acc = stage2_train(net2, trainloader, optimizer, optimizer_criterion,
                                                 criterion, fea_criterion, device)
            save_model(net2, None, epoch, os.path.join(args.checkpoint, 'stage_2_last_model.pth'))
            logger.append([epoch + 1, optimizer.param_groups[0]['lr'], train_loss, train_acc])
            pass_centroids(net2, fea_criterion, init_centroids=None)
            if epoch % 5 ==0:
                test(net2, testloader, device)
        print(f"\nFinish Stage-2 training...\n")
    logger.close()



    test(net2, testloader, device)
    return net2



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
        if k.startswith("classifier"):
            continue    # we do not load the classifier weight from stage 1.
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
    for k,_ in net2_dict.items():
        if k.__contains__('classifier.centroids'):
            net2_dict[k] = centroids

    net2.load_state_dict(net2_dict)



def test( net,  testloader, device):
    net.eval()
    scores, labels = [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs,_,_,_ = net(inputs)
            scores.append(outputs)
            labels.append(targets)
            progress_bar(batch_idx, len(testloader))

    scores = torch.cat(scores, dim=0)
    scores = scores.softmax(dim=1)
    scores = scores.cpu().numpy()

    print(scores.shape)
    labels = torch.cat(labels, dim=0).cpu().numpy()
    pred=[]
    for score in scores:
        pred.append(np.argmax(score) if np.max(score) >= args.oltr_threshold else args.train_class_num)
    eval = Evaluation(pred, labels, scores)
    torch.save(eval, os.path.join(args.checkpoint, 'eval.pkl'))
    print(f"Center-Loss accuracy is %.3f" % (eval.accuracy))
    print(f"Center-Loss F1 is %.3f" % (eval.f1_measure))
    print(f"Center-Loss f1_macro is %.3f" % (eval.f1_macro))
    print(f"Center-Loss f1_macro_weighted is %.3f" % (eval.f1_macro_weighted))
    print(f"Center-Loss area_under_roc is %.3f" % (eval.area_under_roc))


def save_model(net, acc, epoch, path):
    state = {
        'net': net.state_dict(),
        'testacc': acc,
        'epoch': epoch,
    }
    torch.save(state, path)

if __name__ == '__main__':
    main()

