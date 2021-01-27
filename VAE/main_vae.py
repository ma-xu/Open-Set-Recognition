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
from backbones import VanillaVAE
from datasets import MNIST

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
from Utils import mkdir_p, Logger, progress_bar, save_model, save_binary_img


parser = argparse.ArgumentParser(description='VAE training for NSF project')

# General MODEL parameters
parser.add_argument('--latent_dim', default=128, type=int)

# Dataset preperation
parser.add_argument('--train_class_num', default=7, type=int, help='Classes used in training')
parser.add_argument('--test_class_num', default=10, type=int, help='Classes used in testing')
parser.add_argument('--includes_all_train_class', default=True, action='store_true',
                    help='If required all known classes included in testing')


# Parameters for  training
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--small', action='store_true', help='Showcase on small set')
parser.add_argument('--es', default=50, type=int, help='epoch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--bs', default=144, type=int, help='batch size, better to have a square number')
parser.add_argument('--wd', default=0.0, type=float, help='weight decay')
parser.add_argument('--scheduler_gamma', default=0.95, type=float, help='weight decay')

parser.add_argument('--evaluate', action='store_true', help='Evaluate model, ensuring the resume path is given')
parser.add_argument('--val_num', default=8, type=int,
                    help=' the number of validate images, also nrow for saving images')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.checkpoint = './checkpoints/vae/mnist/vanilla_vae-%s-%s-%s' % (
    args.train_class_num, args.test_class_num, args.latent_dim)
if not os.path.isdir(args.checkpoint):
    mkdir_p(args.checkpoint)

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
valloader = torch.utils.data.DataLoader(testset, batch_size=args.val_num, shuffle=False, num_workers=4)

M_N = args.bs / trainset.__len__()


def main():
    start_epoch = 0
    best_loss = 9999999.99

    # Model
    print('==> Building model..')
    net = VanillaVAE(in_channels=1, latent_dim=args.latent_dim)
    net = net.to(device)

    if device == 'cuda':
        # Considering the data scale and model, it is unnecessary to use DistributedDataParallel
        # which could speed up the training and inference compared to DataParallel
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)

    if args.resume:
        # Load checkpoint.
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), resume=True)
            print('==> Resuming from checkpoint, loaded..')
        else:
            print("==> No checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Recons Loss', 'KLD Loss'])

    if not args.evaluate:
        # training
        print("==> start training..")
        for epoch in range(start_epoch, args.es):
            print('\nStage_1 Epoch: %d | Learning rate: %f ' % (epoch + 1, scheduler.get_last_lr()[-1]))
            train_out = train(net, trainloader, optimizer)  # {train_loss, recons_loss, kld_loss}
            save_model(net, optimizer, epoch, os.path.join(args.checkpoint, 'checkpoint.pth'))
            if train_out["train_loss"] < best_loss:
                save_model(net, optimizer, epoch, os.path.join(args.checkpoint, 'checkpoint_best.pth'),
                           loss=train_out["train_loss"])
                best_loss = train_out["train_loss"]
            logger.append([epoch + 1, scheduler.get_last_lr()[-1],
                           train_out["train_loss"], train_out["recons_loss"], train_out["kld_loss"]])
            scheduler.step()
        logger.close()
        print(f"\n==> Finish training..\n")

    print("===> start evaluating ...")
    generate_images(net, valloader, name="test_reconstruct")
    sample_images(net, name="test_randsample")


def train(net, trainloader, optimizer):
    net.train()
    train_loss = 0
    recons_loss = 0
    kld_loss = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        print(f"max: {inputs.max()} min {inputs.min()}")
        optimizer.zero_grad()
        result = net(inputs)
        loss_dict = net.module.loss_function(*result, M_N=M_N)  # loss, Reconstruction_Loss, KLD
        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        recons_loss += (loss_dict['Reconstruction_Loss']).item()
        kld_loss += (loss_dict['KLD']).item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Rec_Loss: %.3f | KLD_Loss: %.3f'
                     % (train_loss / (batch_idx + 1), recons_loss / (batch_idx + 1), kld_loss / (batch_idx + 1)))

    return {
        "train_loss": train_loss / (batch_idx + 1),
        "recons_loss": recons_loss / (batch_idx + 1),
        "kld_loss": kld_loss / (batch_idx + 1)
    }


def generate_images(net, valloader, name="val"):
    dataloader_iterator = iter(valloader)
    with torch.no_grad():
        img, spe = next(dataloader_iterator)
        img = img.to(device)
        recons = net.module.generate(img)
        result = torch.cat([img, recons], dim=0)
        save_binary_img(result.data,
                        os.path.join(args.checkpoint, f"{name}.png"),
                        nrow=args.val_num)


def sample_images(net, name="val"):
    with torch.no_grad():
        z = torch.randn(args.val_num, args.latent_dim)
        z = z.to(device)
        sampled = net.module.sample(z)
        save_binary_img(sampled.data,
                        os.path.join(args.checkpoint, f"{name}.png"),
                        nrow=args.val_num)


if __name__ == '__main__':
    main()
