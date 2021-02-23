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
from DFPLoss import DFPLoss, DFPNormLoss
from DFPNet import DFPNet
from MyPlotter import plot_feature
from energy_hist import plot_listhist

# python3 cifar100.py --temperature 1 --hist_save

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
parser.add_argument('--p', default=2, type=int, help='p-norm')

# Parameters for optimizer
parser.add_argument('--temperature', default=1, type=int, help='scaling cosine distance for exp')
parser.add_argument('--alpha', default=0.5, type=float, help='balance for classfication and energy loss')

# Parameters for stage 1 training
parser.add_argument('--stage1_resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--stage1_es', default=100, type=int, help='epoch size')
parser.add_argument('--stage1_lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--stage1_lr_factor', default=0.1, type=float, help='learning rate Decay factor')  # works for MNIST
parser.add_argument('--stage1_lr_step', default=30, type=float, help='learning rate Decay step')  # works for MNIST
parser.add_argument('--stage1_bs', default=128, type=int, help='batch size')
parser.add_argument('--evaluate', action='store_true', help='Evaluate without training')

# parameters for mixup
parser.add_argument('--mixup', default=1., type=float, help='the parameters for mixup')

# Parameters for stage plotting
parser.add_argument('--plot', action='store_true', help='Plotting the training set.')
parser.add_argument('--plot_quality', default=200, type=int, help='DPI of plot figure')

# histogram figures for Energy model analysis
parser.add_argument('--hist_bins', default=100, type=int, help='divided into n bins')
parser.add_argument('--hist_norm', default=True, action='store_true', help='if norm the frequency to [0,1]')
parser.add_argument('--hist_save', action='store_true', help='if save the histogram figures')

# Parameters for stage 2 training
parser.add_argument('--stage2_resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--stage2_es', default=50, type=int, help='epoch size')
parser.add_argument('--stage2_lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--stage2_lr_factor', default=0.1, type=float, help='learning rate Decay factor')  # works for MNIST
parser.add_argument('--stage2_lr_step', default=20, type=float, help='learning rate Decay step')  # works for MNIST
parser.add_argument('--stage2_bs', default=128, type=int, help='batch size')


args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.checkpoint = './checkpoints/cifar10/%s_%s_%s_dim%s_T%s_alpha%s_p%s' % (
    args.train_class_num, args.test_class_num, args.arch, args.embed_dim, args.temperature,args.alpha,args.p)
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
trainset = CIFAR10(root='../../data', train=True, download=True, transform=transform_train,
                    train_class_num=args.train_class_num, test_class_num=args.test_class_num,
                    includes_all_train_class=args.includes_all_train_class)
testset = CIFAR10(root='../../data', train=False, download=True, transform=transform_test,
                   train_class_num=args.train_class_num, test_class_num=args.test_class_num,
                   includes_all_train_class=args.includes_all_train_class)
# data loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.stage1_bs, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.stage1_bs, shuffle=False, num_workers=4)


def main():
    print(device)
    stage1_dict={
        "net":None,
        "mid_known": None,
        "mid_unknown": None
    }
    if not args.stage2_resume:
        stage1_dict = main_stage1()  # {"net": net, "mid_known","mid_unknown"}
    main_stage2(stage1_dict)


def main_stage1():
    print(f"\nStart Stage-1 training ...\n")
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    print('==> Building model..')
    net = DFPNet(backbone=args.arch, num_classes=args.train_class_num, embed_dim=args.embed_dim, p=args.p)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = DFPLoss(temperature=args.temperature)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.stage1_lr, momentum=0.9, weight_decay=5e-4)

    if args.stage1_resume:
        # Load checkpoint.
        if os.path.isfile(args.stage1_resume):
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(args.stage1_resume)
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            logger = Logger(os.path.join(args.checkpoint, 'log_stage1.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log_stage1.txt'))
        logger.set_names(['Epoch', 'Train Loss', 'Train Acc.'])

    if not args.evaluate:
        for epoch in range(start_epoch, args.stage1_es):
            adjust_learning_rate(optimizer, epoch, args.stage1_lr,
                                 factor=args.stage1_lr_factor, step=args.stage1_lr_step)
            print('\nStage_1 Epoch: %d | Learning rate: %f ' % (epoch + 1, optimizer.param_groups[0]['lr']))
            train_out = stage1_train(net, trainloader, optimizer, criterion, device)
            save_model(net, optimizer, epoch, os.path.join(args.checkpoint, 'stage_1_last_model.pth'))
            logger.append([epoch + 1, train_out["train_loss"], train_out["accuracy"]])
            if args.plot:
                plot_feature(net, args, trainloader, device, args.plotfolder, epoch=epoch,
                             plot_class_num=args.train_class_num, plot_quality=args.plot_quality)
                plot_feature(net, args, testloader, device, args.plotfolder, epoch="test" + str(epoch),
                             plot_class_num=args.train_class_num + 1, plot_quality=args.plot_quality, testmode=True)
        logger.close()
        print(f"\nFinish Stage-1 training...\n")

    print("===> Evaluating stage-1 ...")
    stage_test(net, testloader, device)
    mid_dict = stage_valmixup(net, trainloader, device)
    return {
        "net": net.state_dict(),
        "mid_known": mid_dict["mid_known"],
        "mid_unknown": mid_dict["mid_unknown"]
    }


def stage1_train(net, trainloader, optimizer, criterion, device):
    net.train()
    train_loss = 0
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

        _, predicted = (out['normweight_fea2cen']).max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return {
        "train_loss": train_loss / (batch_idx + 1),
        "accuracy": correct / total
    }


def main_stage2(stage1_dict):
    print("Starting stage-2 fine-tuning ...")
    start_epoch = 0

    # get key values from stage1_dict
    mid_known = stage1_dict["mid_known"]
    mid_unknown = stage1_dict["mid_unknown"]
    net_state_dict = stage1_dict["net"]

    net = DFPNet(backbone=args.arch, num_classes=args.train_class_num, embed_dim=args.embed_dim, p=args.p)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    optimizer = torch.optim.SGD(net.parameters(), lr=args.stage2_lr, momentum=0.9, weight_decay=5e-4)
    if args.stage2_resume:
        # Load checkpoint.
        if os.path.isfile(args.stage2_resume):
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(args.stage2_resume)
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            mid_known =checkpoint["mid_known"]
            mid_unknown =checkpoint["mid_unknown"]
            logger = Logger(os.path.join(args.checkpoint, 'log_stage2.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        net.load_state_dict(net_state_dict)
        logger = Logger(os.path.join(args.checkpoint, 'log_stage2.txt'))
        logger.set_names(['Epoch', 'Train Loss', 'Class Loss', 'Energy Loss','Energy Known', 'Energy Unknown', 'Train Acc.'])

    criterion = DFPNormLoss(mid_known=1.3*mid_known, mid_unknown=0.7*mid_unknown,
                            alpha=args.alpha, temperature=args.temperature, feature='energy')

    if not args.evaluate:
        for epoch in range(start_epoch, args.stage2_es):
            adjust_learning_rate(optimizer, epoch, args.stage2_lr,
                                 factor=args.stage2_lr_factor, step=args.stage2_lr_step)
            print('\nStage_2 Epoch: %d | Learning rate: %f ' % (epoch + 1, optimizer.param_groups[0]['lr']))
            train_out = stage2_train(net, trainloader, optimizer, criterion, device)
            save_model(net, optimizer, epoch,
                       os.path.join(args.checkpoint, 'stage_2_last_model.pth'),
                       mid_known=mid_known, mid_unknown=mid_unknown)
            logger.append([epoch + 1, train_out["train_loss"], train_out["loss_classification"],
                           train_out["loss_energy"], train_out["loss_energy_known"],
                           train_out["loss_energy_unknown"], train_out["accuracy"]])
            if args.plot:
                plot_feature(net, args, trainloader, device, args.plotfolder, epoch="stage2_"+str(epoch),
                             plot_class_num=args.train_class_num, plot_quality=args.plot_quality)
                plot_feature(net, args, testloader, device, args.plotfolder, epoch="stage2_test" + str(epoch),
                             plot_class_num=args.train_class_num + 1, plot_quality=args.plot_quality, testmode=True)
        logger.close()
        print(f"\nFinish Stage-2 training...\n")

        print("===> Evaluating stage-2 ...")
        stage_test(net, testloader, device, name="stage2_test_doublebar")
        stage_valmixup(net, trainloader, device, name="stage2_mixup_result")
        stage_evaluate(net, testloader, mid_unknown, mid_known, feature="energy")


# Training
def stage2_train(net, trainloader, optimizer, criterion, device):
    net.train()
    train_loss = 0
    loss_classification = 0
    loss_energy = 0
    loss_energy_known = 0
    loss_energy_unknown = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        mixed = mixup(inputs, targets, args)
        optimizer.zero_grad()
        out = net(inputs)
        out_unkown = net(mixed)
        loss_dict = criterion(out, targets, out_unkown)
        loss = loss_dict['total']
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loss_classification += (loss_dict['loss_classification']).item()
        loss_energy += (loss_dict['loss_energy']).item()
        loss_energy_known += (loss_dict['loss_energy_known']).item()
        loss_energy_unknown += (loss_dict['loss_energy_unknown']).item()

        _, predicted = (out['normweight_fea2cen']).max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return {
        "train_loss": train_loss / (batch_idx + 1),
        "loss_classification": loss_classification / (batch_idx + 1),
        "loss_energy": loss_energy / (batch_idx + 1),
        "loss_energy_known": loss_energy_known / (batch_idx + 1),
        "loss_energy_unknown": loss_energy_unknown / (batch_idx + 1),
        "accuracy": correct / total
    }





def stage_test(net, testloader, device, name="stage1_test_doublebar"):
    correct = 0
    total = 0
    normfea_list = []
    pnorm_list = []
    energy_list = []
    normweight_fea2cen_list = []
    Target_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            out = net(inputs)  # shape [batch,class]
            normfea_list.append(out["norm_fea"])
            pnorm_list.append(out["pnorm"])
            energy_list.append(out["energy"])
            normweight_fea2cen_list.append(out["normweight_fea2cen"])
            Target_list.append(targets)
            _, predicted = (out["normweight_fea2cen"]).max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), '| Acc: %.3f%% (%d/%d)'
                         % (100. * correct / total, correct, total))
    print("\nTesting results is {:.2f}%".format(100. * correct / total))

    normfea_list = torch.cat(normfea_list, dim=0)
    pnorm_list = torch.cat(pnorm_list, dim=0)
    energy_list = torch.cat(energy_list, dim=0)
    normweight_fea2cen_list = torch.cat(normweight_fea2cen_list, dim=0)
    softmax_list = torch.softmax(normweight_fea2cen_list, dim=1).max(dim=1, keepdim=False)[0]
    Target_list = torch.cat(Target_list, dim=0)
    unknown_label = Target_list.max()
    unknown_normfea_list = normfea_list[Target_list == unknown_label]
    known_normfea_list = normfea_list[Target_list != unknown_label]

    unknown_pnorm_list = pnorm_list[Target_list == unknown_label]
    known_pnorm_list = pnorm_list[Target_list != unknown_label]

    unknown_energy_list = energy_list[Target_list == unknown_label]
    known_energy_list = energy_list[Target_list != unknown_label]

    unknown_softmax_list = softmax_list[Target_list == unknown_label]
    known_softmax_list = softmax_list[Target_list != unknown_label]


    print("_______________Testing statistics:____________")
    print(f"test known mid:{known_energy_list.median()} | unknown mid:{unknown_energy_list.median()}")
    print(f"min  energy:{min(known_energy_list.min(), unknown_energy_list.min())} "
          f"| max  energy:{max(known_energy_list.max(), unknown_energy_list.max())}")
    plot_listhist([known_normfea_list, unknown_normfea_list],
                  args, labels=["known", "unknown"],
                  name=name+"_normfea")
    plot_listhist([known_pnorm_list, unknown_pnorm_list],
                  args, labels=["known", "unknown"],
                  name=name+"_pnorm")

    plot_listhist([known_energy_list, unknown_energy_list],
                  args, labels=["known", "unknown"],
                  name=name + "_energy")

    plot_listhist([known_softmax_list, unknown_softmax_list],
                  args, labels=["known", "unknown"],
                  name=name + "_softmax")


def stage_valmixup(net, dataloader, device, name="stage1_mixup_doublebar"):
    print("validating mixup and trainloader ...")
    normfea_loader_list = []
    normfea_mixup_list = []
    pnorm_loader_list = []
    pnorm_mixup_list = []
    energy_loader_list = []
    energy_mixup_list = []
    normweight_loader_list = []
    normweight_mixup_list = []
    target_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            mixed = mixup(inputs, targets, args)
            out_loader = net(inputs)
            out_mixed = net(mixed)
            normfea_loader_list.append(out_loader["norm_fea"])
            normfea_mixup_list.append(out_mixed["norm_fea"])
            pnorm_loader_list.append(out_loader["pnorm"])
            pnorm_mixup_list.append(out_mixed["pnorm"])
            energy_loader_list.append(out_loader["energy"])
            energy_mixup_list.append(out_mixed["energy"])
            normweight_loader_list.append(out_loader["normweight_fea2cen"])
            normweight_mixup_list.append(out_mixed["normweight_fea2cen"])


            target_list.append(targets)
            progress_bar(batch_idx, len(trainloader))

    normfea_loader_list = torch.cat(normfea_loader_list, dim=0)
    normfea_mixup_list = torch.cat(normfea_mixup_list, dim=0)
    pnorm_loader_list = torch.cat(pnorm_loader_list,dim=0)
    pnorm_mixup_list = torch.cat(pnorm_mixup_list,dim=0)
    energy_loader_list = torch.cat(energy_loader_list, dim=0)
    energy_mixup_list = torch.cat(energy_mixup_list, dim=0)
    normweight_loader_list = torch.cat(normweight_loader_list,dim=0)
    softmax_loader_list = torch.softmax(normweight_loader_list, dim=1).max(dim=1, keepdim=False)[0]
    normweight_mixup_list = torch.cat(normweight_mixup_list, dim=0)
    softmax_mixup_list = torch.softmax(normweight_mixup_list, dim=1).max(dim=1, keepdim=False)[0]


    plot_listhist([pnorm_loader_list, pnorm_mixup_list],
                  args, labels=["loader", "mixup"],
                  name=name + "_pnorm")

    plot_listhist([normfea_loader_list, normfea_mixup_list],
                  args, labels=["loader", "mixup"],
                  name=name + "_normfea")

    plot_listhist([energy_loader_list, energy_mixup_list],
                  args, labels=["loader", "mixup"],
                  name=name + "_energy")

    plot_listhist([softmax_loader_list, softmax_mixup_list],
                  args, labels=["loader", "mixup"],
                  name=name + "_softmax")

    print("_______________Validate statistics:____________")
    print(f"train mid:{energy_loader_list.median()} | mixup mid:{energy_mixup_list.median()}")
    print(f"min  energy:{min(energy_loader_list.min(), energy_mixup_list.min())} "
          f"| max  energy:{max(energy_loader_list.max(), energy_mixup_list.max())}")
    return{
        "mid_known": energy_loader_list.median(),
        "mid_unknown": energy_mixup_list.median()
    }


def stage_evaluate(net,testloader,t_min, t_max, feature='energy'):
    Feature_list = []
    Predict_list = []
    Target_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            out = net(inputs)  # shape [batch,class]
            Feature_list.append(out[feature])
            Target_list.append(targets)
            _, predicted = (out["normweight_fea2cen"]).max(1)
            Predict_list.append(predicted)
            progress_bar(batch_idx, len(testloader), '| ')

    Feature_list = torch.cat(Feature_list, dim=0)
    Target_list = torch.cat(Target_list, dim=0)
    Predict_list = torch.cat(Predict_list, dim=0)

    best_thres = 0.
    best_eval = None
    best_f1_measure = 0.
    for thres in np.linspace(t_min,t_max,20):
        Predict_list[Feature_list<thres] = args.train_class_num
        eval = Evaluation(Predict_list.cpu().numpy(),Target_list.cpu().numpy(),Feature_list.cpu().numpy())
        if eval.f1_measure >best_f1_measure:
            best_thres = thres
            best_eval = eval
    print("===> Finial Evaluation...")
    print(f"threshold is: {best_thres}")
    print(f"F1: {best_eval.f1_measure}\nmacro-F1: {best_eval.f1_macro}")



def mixup(inputs, targets, args):
    shuffle = torch.randperm(inputs.shape[0]).to(inputs.device)
    inputs_bak = inputs[shuffle]
    targets_bak = targets[shuffle]
    dis_matchers = ~targets.eq(targets_bak)
    mix1 = inputs[dis_matchers]
    mix2 = inputs_bak[dis_matchers]
    lam = np.random.beta(args.mixup, args.mixup)
    lam = max(0.3, min(lam, 0.7))
    mixed = lam * mix1 + (1. - lam) * mix2
    # add Gaussian white Noise for adversarial training
    return mixed
    # noise = torch.randn_like(mixed).to(mixed.device)

    # return 0.8*mixed + 0.2*noise


if __name__ == '__main__':
    main()
