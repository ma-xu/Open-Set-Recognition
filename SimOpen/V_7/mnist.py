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
from backbones import VanillaVAE
from datasets import MNIST
from Utils import adjust_learning_rate, progress_bar, Logger, mkdir_p, Evaluation
from DFPLoss import DFPLoss, DFPEnergyLoss
from DFPNet import DFPNet
from MyPlotter import plot_feature
from energy_hist import energy_hist, energy_hist_sperate

# python3 mnist.py  --hist_save --plot

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
parser.add_argument('--arch', default='LeNetPlus', choices=model_names, type=str, help='choosing network')
parser.add_argument('--embed_dim', default=2, type=int, help='embedding feature dimension')

# Parameters for optimizer
parser.add_argument('--temperature', default=1, type=int, help='scaling cosine distance for exp')
parser.add_argument('--alpha', default=0.1, type=float, help='balance for classfication and energy loss')

# Parameters for stage 1 training
parser.add_argument('--stage1_resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--stage1_es', default=35, type=int, help='epoch size')
parser.add_argument('--stage1_lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--stage1_lr_factor', default=0.1, type=float, help='learning rate Decay factor')  # works for MNIST
parser.add_argument('--stage1_lr_step', default=10, type=float, help='learning rate Decay step')  # works for MNIST
parser.add_argument('--stage1_bs', default=128, type=int, help='batch size')
parser.add_argument('--evaluate', action='store_true', help='Evaluate without training')

# Parameters for stage plotting
parser.add_argument('--plot', action='store_true', help='Plotting the training set.')
parser.add_argument('--plot_quality', default=200, type=int, help='DPI of plot figure')

# histogram figures for Energy model analysis
parser.add_argument('--hist_bins', default=100, type=int, help='divided into n bins')
parser.add_argument('--hist_norm', default=True, action='store_true', help='if norm the frequency to [0,1]')
parser.add_argument('--hist_save', action='store_true', help='if save the histogram figures')
# parser.add_argument('--hist_list', default=["norm_fea","normweight_fea2cen","cosine_fea2cen"],
#                     type=str, nargs='+', help='what outputs to analysis')


# parameters for vae
parser.add_argument('--latent_dim', default=128, type=int)
parser.add_argument('--vae_resume', default='/home/UNT/jg0737/Open-Set-Recognition/VAE/checkpoints/vae/mnist/vanilla_vae-7-10-128/checkpoint_best.pth',
                    type=str, metavar='PATH', help='path to vae checkpoint')

# Parameters for stage 2 training
parser.add_argument('--stage2_resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--stage2_es', default=15, type=int, help='epoch size')
parser.add_argument('--stage2_lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--stage2_lr_factor', default=0.1, type=float, help='learning rate Decay factor')  # works for MNIST
parser.add_argument('--stage2_lr_step', default=6, type=float, help='learning rate Decay step')  # works for MNIST
parser.add_argument('--stage2_bs', default=128, type=int, help='batch size')


args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.checkpoint = './checkpoints/mnist/%s-%s-%s-dim%s-T%s-alpha%s' % (
    args.train_class_num, args.test_class_num, args.arch, args.embed_dim, args.temperature,args.alpha)
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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.stage1_bs, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.stage1_bs, shuffle=False, num_workers=4)


def main():
    print(device)
    stage1_dict = main_stage1()  # {"net": net, "mid_energy": {"mid_known":, "mid_unknown":}}}
    middle_dict = middle_validate(stage1_dict["net"], trainloader, device, stage="1")
    mid_energy = {"mid_known": middle_dict["mid_known"], "mid_unknown": middle_dict["mid_unknown"]}
    main_stage2(stage1_dict["net"],middle_dict["vae"], mid_energy)


def main_stage1():
    print(f"\nStart Stage-1 training ...\n")
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    print('==> Building model..')
    net = DFPNet(backbone=args.arch, num_classes=args.train_class_num, embed_dim=args.embed_dim)
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
    stage1_test(net, testloader, device)

    return {
        "net": net,
    }


# Training
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


def stage1_test(net, testloader, device):
    correct = 0
    total = 0
    normweight_fea2cen_list = []
    Target_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            out = net(inputs)  # shape [batch,class]
            normweight_fea2cen_list.append(out["normweight_fea2cen"])
            Target_list.append(targets)
            _, predicted = (out["normweight_fea2cen"]).max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), '| Acc: %.3f%% (%d/%d)'
                         % (100. * correct / total, correct, total))
    print("\nTesting results is {:.2f}%".format(100. * correct / total))

    normweight_fea2cen_list = torch.cat(normweight_fea2cen_list, dim=0)
    Target_list = torch.cat(Target_list, dim=0)

    logsumexp_result = args.temperature * \
                                     torch.logsumexp(normweight_fea2cen_list / args.temperature, dim=1, keepdim=False)
    max_result = torch.max(normweight_fea2cen_list, dim=1, keepdim=False)[0]
    softmax_result = torch.softmax(normweight_fea2cen_list,dim=1).max(dim=1, keepdim=False)[0]


    scaled_ = (normweight_fea2cen_list - normweight_fea2cen_list.min())\
              /(normweight_fea2cen_list.max()-normweight_fea2cen_list.min())
    smoothmaximum_factor = torch.exp(1.0 * scaled_)
    smoothmaximum_result = (normweight_fea2cen_list*smoothmaximum_factor).sum(dim=1, keepdim=False) \
                          / smoothmaximum_factor.sum(dim=1, keepdim=False)
    p4norm_result = normweight_fea2cen_list.norm(p=4,dim=1,keepdim=False)
    p3norm_result = normweight_fea2cen_list.norm(p=3, dim=1, keepdim=False)
    p2norm_result = normweight_fea2cen_list.norm(p=2, dim=1, keepdim=False)
    p1norm_result = normweight_fea2cen_list.norm(p=1, dim=1, keepdim=False)
    p5norm_result = normweight_fea2cen_list.norm(p=5, dim=1, keepdim=False)

    energy_hist(normweight_fea2cen_list, Target_list, args, "logits_result")
    energy_hist(logsumexp_result, Target_list, args, "logsumexp_result")
    energy_hist(max_result, Target_list, args, "max_result")
    energy_hist(softmax_result, Target_list, args, "softmax_result")
    energy_hist(smoothmaximum_result, Target_list, args, "smoothmaximum_result")
    energy_hist(p1norm_result, Target_list, args, "p1norm_result")
    energy_hist(p2norm_result, Target_list, args, "p2norm_result")
    energy_hist(p3norm_result, Target_list, args, "p3norm_result")
    energy_hist(p4norm_result, Target_list, args, "p4norm_result")
    energy_hist(p5norm_result, Target_list, args, "p5norm_result")


def middle_validate(net, trainloader, device, stage="1"):
    print("validating vae and net ...")
    known_energy, unknown_energy = [], []

    # loading vae model
    vae = VanillaVAE(in_channels=1,latent_dim = args.latent_dim)
    vae = vae.to(device)
    if device == 'cuda':
        vae = torch.nn.DataParallel(vae)
        cudnn.benchmark = True
    if os.path.isfile(args.vae_resume):
        vae_checkpoint = torch.load(args.vae_resume)
        vae.load_state_dict(vae_checkpoint['net'])
        print('==> Resuming vae from checkpoint, loaded..')

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            sampled = sampler(vae,device,args)
            out_known = net(inputs)
            out_unkown = net(sampled)
            known_energy.append(out_known["energy"])
            unknown_energy.append(out_unkown["energy"])
            progress_bar(batch_idx, len(trainloader))

    known_energy = torch.cat(known_energy, dim=0)
    unknown_energy = torch.cat(unknown_energy, dim=0)
    energy_hist_sperate(known_energy, unknown_energy, args, "mixup_stage"+stage)
    return{
        # unkown is smaller than known
        "vae": vae,
        "mid_known": known_energy.median().data,
        "mid_unknown": unknown_energy.median().data
    }


def main_stage2(net, vae, mid_energy):
    print("Starting stage-2 fine-tuning ...")
    start_epoch = 0
    criterion = DFPEnergyLoss(mid_known=mid_energy["mid_known"], mid_unknown=mid_energy["mid_unknown"],
                              alpha=args.alpha, temperature=args.temperature)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.stage2_lr, momentum=0.9, weight_decay=5e-4)
    if args.stage2_resume:
        # Load checkpoint.
        if os.path.isfile(args.stage2_resume):
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(args.stage2_resume)
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            logger = Logger(os.path.join(args.checkpoint, 'log_stage2.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log_stage2.txt'))
        logger.set_names(['Epoch', 'Train Loss', 'Class Loss', 'Energy Loss','Energy Known', 'Energy Unknown', 'Train Acc.'])

    if not args.evaluate:
        for epoch in range(start_epoch, args.stage2_es):
            adjust_learning_rate(optimizer, epoch, args.stage2_lr,
                                 factor=args.stage2_lr_factor, step=args.stage2_lr_step)
            print('\nStage_2 Epoch: %d | Learning rate: %f ' % (epoch + 1, optimizer.param_groups[0]['lr']))
            train_out = stage2_train(net, trainloader, vae, optimizer, criterion, device)
            save_model(net, optimizer, epoch, os.path.join(args.checkpoint, 'stage_2_last_model.pth'))
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
        stage2_test(net, testloader, trainloader, device)


# Training
def stage2_train(net, trainloader,vae, optimizer, criterion, device):
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
        sampled = sampler(vae, device, args)
        optimizer.zero_grad()
        out = net(inputs)
        out_unkown = net(sampled)
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


def stage2_test(net, testloader, trainloader, device ):
    correct = 0
    total = 0
    energy_list = []
    Target_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            out = net(inputs)  # shape [batch,class]
            energy_list.append(out["energy"])
            Target_list.append(targets)
            _, predicted = (out["normweight_fea2cen"]).max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), '| Acc: %.3f%% (%d/%d)'
                         % (100. * correct / total, correct, total))
    print("\nTesting results is {:.2f}%".format(100. * correct / total))

    energy_list = torch.cat(energy_list, dim=0)
    Target_list = torch.cat(Target_list, dim=0)
    energy_hist(energy_list, Target_list, args, "testing_energy")

    middle_validate(net, trainloader, device, stage="2")


def save_model(net, optimizer, epoch, path, **kwargs):
    state = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    for key, value in kwargs.items():
        state[key] = value
    torch.save(state, path)


def mixup(inputs, targets, inputs_bak, targets_bak, args):
    dis_matchers = ~targets.eq(targets_bak)
    mix1 = inputs[dis_matchers]
    mix2 = inputs_bak[dis_matchers]
    lam = np.random.beta(args.mixup, args.mixup)
    lam = max(0.1, min(lam, 0.9))
    mixed = lam * mix1 + (1. - lam) * mix2
    return mixed


def sampler(vae, device, args):
    z = torch.randn(args.stage1_bs, args.latent_dim)
    z = z.to(device)
    sampled = vae.module.sample(z)
    sampled = sampled.detach()
    return sampled.sub_(0.1307).div_(0.3081)

if __name__ == '__main__':
    main()
