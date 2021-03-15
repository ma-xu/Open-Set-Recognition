### The Released Code for "Counterfactual Zero-shot and Open-Set Visual Recognition"
### Author: Wang Tan
### Part of Code borrow from "CGDL"

from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import argparse
import os
import time
import utils
from dataloader import MNIST_Dataset, CIFAR10_Dataset, SVHN_Dataset, CIFARAdd10_Dataset, CIFARAdd50_Dataset, CIFARAddN_Dataset
#from keras.utils import to_categorical
from model import LVAE
from qmv import ocr_test

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch OSR Example')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 1e-3)')
    parser.add_argument('--wd', type=float, default=0.00, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.01, help='momentum (default: 1e-3)')
    parser.add_argument('--decreasing_lr', default='60,100,150', help='decreasing strategy')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='decreasing strategy')
    parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
    parser.add_argument('--seed_sampler', type=str, default='777 1234 2731 3925 5432', help='random seed for dataset sampler')
    parser.add_argument('--log_interval', type=int, default=20,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--val_interval', type=int, default=5, help='how many epochs to wait before another val')
    parser.add_argument('--test_interval', type=int, default=5, help='how many epochs to wait before another test')
    parser.add_argument('--lamda', type=int, default=100, help='lamda in loss function')
    parser.add_argument('--beta_z', type=int, default=1, help='beta of the kl in loss function')
    parser.add_argument('--beta_anneal', type=int, default=0, help='the anneal epoch of beta')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold of gaussian model')
    parser.add_argument('--tensorboard', action="store_true", default=False, help='If use tensorboard')
    parser.add_argument('--debug', action="store_true", default=False, help='If debug mode')

    # train
    parser.add_argument('--dataset', type=str, default="MNIST", help='The dataset going to use')
    parser.add_argument('--eval', action="store_true", default=False, help='directly eval?')
    parser.add_argument('--baseline', action="store_true", default=False, help='If is the bseline?')
    parser.add_argument('--use_model', action="store_true", default=False, help='If use model to get the train feature')
    parser.add_argument('--encode_z', type=int, default=None, help='If encode z and dim of z')
    parser.add_argument("--contrastive_loss", action="store_true", default=False, help="Use contrastive loss")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for contrastive loss")
    parser.add_argument("--contra_lambda", type=float, default=1.0, help="Scaling factor of contrastive loss")
    parser.add_argument("--save_epoch", type=int, default=None, help="save model in this epoch")
    parser.add_argument("--exp", type=int, default=0, help="which experiment")
    parser.add_argument("--unseen_num", type=int, default=13, help="unseen class num in CIFAR100")

    # test
    parser.add_argument('--cf', action="store_true", default=False, help='use counterfactual generation')
    parser.add_argument('--cf_threshold', action="store_true", default=False, help='use counterfactual threshold in revise_cf')
    parser.add_argument('--yh', action="store_true", default=False, help='use yh rather than feature_y_mean')
    parser.add_argument('--use_model_gau', action="store_true", default=False, help='use feature by model in gau')


    args = parser.parse_args()
    return args

def control_seed(args):
    # seed
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True


class DeterministicWarmup(object):
    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1 / n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t  # 0->1
        return self.t


def train(args, lvae):
    best_val_loss = 1000
    # train
    for epoch in range(args.epochs):
        lvae.train()
        print("Training... Epoch = %d" % epoch)
        correct_train = 0

        if args.beta_anneal != 0:
            args.beta_z = next(args.beta_anneal)

        open('%s/train_fea.txt' % args.save_path, 'w').close()
        open('%s/train_tar.txt' % args.save_path, 'w').close()
        open('%s/train_rec.txt' % args.save_path, 'w').close()

        if epoch in decreasing_lr:
            optimizer.param_groups[0]['lr'] *= args.lr_decay
            print("~~~learning rate:", optimizer.param_groups[0]['lr'])
        for batch_idx, (data, target) in enumerate(train_loader):
            target_en = torch.Tensor(target.shape[0], args.num_classes)
            target_en.zero_()
            target_en.scatter_(1, target.view(-1, 1), 1)  # one-hot encoding
            target_en = target_en.to(device)
            if args.cuda:
                data = data.cuda()
                target = target.cuda()
            data, target = Variable(data), Variable(target)

            loss, mu, output, output_mu, x_re, rec, kl, ce = lvae.loss(data, target, target_en, next(beta), args.lamda, args)
            rec_loss = (x_re - data).pow(2).sum((3, 2, 1))
            if args.contrastive_loss:
                contra_loss = lvae.contra_loss
                print_rec = contra_loss + rec
            else:
                print_rec = rec

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            outlabel = output.data.max(1)[1]  # get the index of the max log-probability
            correct_train += outlabel.eq(target.view_as(outlabel)).sum().item()

            cor_fea = mu[(outlabel == target)]
            cor_tar = target[(outlabel == target)]
            cor_fea = torch.Tensor.cpu(cor_fea).detach().numpy()
            cor_tar = torch.Tensor.cpu(cor_tar).detach().numpy()
            rec_loss = torch.Tensor.cpu(rec_loss).detach().numpy()

            with open('%s/train_fea.txt' % args.save_path, 'ab') as f:
                np.savetxt(f, cor_fea, fmt='%f', delimiter=' ', newline='\r')
                f.write(b'\n')
            with open('%s/train_tar.txt' % args.save_path, 'ab') as t:
                np.savetxt(t, cor_tar, fmt='%d', delimiter=' ', newline='\r')
                t.write(b'\n')
            with open('%s/train_rec.txt' % args.save_path, 'ab') as m:
                np.savetxt(m, rec_loss, fmt='%f', delimiter=' ', newline='\r')
                m.write(b'\n')

            if batch_idx % args.log_interval == 0:
                print('[Run {}] Train Epoch: {} [{}/{} ({:.0f}%)]  lr:{}  loss:{:.3f} = rec:{:.3f} + kl:{:.3f} + ce:{:.3f}'.format(
                    args.run_idx, epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx * len(data) / len(train_loader.dataset),
                           optimizer.param_groups[0]['lr'],
                           loss.data / (len(data)),
                           print_rec.data / (len(data)),
                           kl.data / (len(data)),
                           ce.data / (len(data))
                    ))

        train_acc = float(100 * correct_train) / len(train_loader.dataset)
        print('Train_Acc: {}/{} ({:.2f}%)'.format(correct_train, len(train_loader.dataset), train_acc))

        # write into the tensorboard
        if args.tensorboard:
            writer.add_scalar("loss_all", loss.data/len(data), epoch)
            writer.add_scalar("rec_loss", rec.data/ len(data), epoch)
            writer.add_scalar("kl_loss", kl.data/len(data), epoch)
            writer.add_scalar("ce_loss", ce.data/len(data), epoch)
            writer.add_scalar("Acc_train", train_acc, epoch)
            if args.contrastive_loss:
                writer.add_scalar("contra_loss", contra_loss.data / len(data), epoch)


        # val on the val set
        if epoch % args.val_interval == 0 and epoch >= 0:
            lvae.eval()
            correct_val = 0
            total_val_loss = 0
            total_val_rec = 0
            total_val_kl = 0
            total_val_ce = 0
            for data_val, target_val in val_loader:
                target_val_en = torch.Tensor(target_val.shape[0], args.num_classes)
                target_val_en.zero_()
                target_val_en.scatter_(1, target_val.view(-1, 1), 1)  # one-hot encoding
                target_val_en = target_val_en.to(device)
                if args.cuda:
                    data_val, target_val = data_val.cuda(), target_val.cuda()
                with torch.no_grad():
                    data_val, target_val = Variable(data_val), Variable(target_val)

                loss_val, mu_val, output_val, output_mu_val, val_re, rec_val, kl_val, ce_val = lvae.loss(data_val, target_val, target_val_en, next(beta), args.lamda, args)
                total_val_loss += loss_val.data.detach().item()
                total_val_rec += rec_val.data.detach().item()
                total_val_kl += kl_val.data.detach().item()
                total_val_ce += ce_val.data.detach().item()

                vallabel = output_val.data.max(1)[1]  # get the index of the max log-probability
                correct_val += vallabel.eq(target_val.view_as(vallabel)).sum().item()

            val_loss = total_val_loss / len(val_loader.dataset)
            val_rec = total_val_rec / len(val_loader.dataset)
            val_kl = total_val_kl / len(val_loader.dataset)
            val_ce = total_val_ce / len(val_loader.dataset)
            print('====> Epoch: {} Val loss: {:.3f}/{} ({:.3f}={:.3f}+{:.3f}+{:.3f})'.format(epoch, total_val_loss, len(val_loader.dataset), val_loss, val_rec, val_kl, val_ce))
            val_acc = float(100 * correct_val) / len(val_loader.dataset)
            print('Val_Acc: {}/{} ({:.2f}%)'.format(correct_val, len(val_loader.dataset), val_acc))

            # write into the tensorboard
            if args.tensorboard:
                writer.add_scalar("val_loss_all", val_loss, epoch)
                writer.add_scalar("val_rec_loss", val_rec, epoch)
                writer.add_scalar("val_kl_loss", val_kl, epoch)
                writer.add_scalar("val_ce_loss", val_ce, epoch)
                writer.add_scalar("Acc_val", val_acc, epoch)


            ## if val best
            if val_loss < best_val_loss or (args.save_epoch != None and epoch == args.save_epoch):
                # save model
                states = {}
                states['epoch'] = epoch
                states['model'] = lvae.state_dict()
                states['val_loss'] = val_loss
                torch.save(states, os.path.join(args.save_path, 'model.pkl'))

                best_val_loss = val_loss
                best_val_epoch = epoch


                train_fea = np.loadtxt('%s/train_fea.txt' % args.save_path)
                train_tar = np.loadtxt('%s/train_tar.txt' % args.save_path)
                train_rec = np.loadtxt('%s/train_rec.txt' % args.save_path)

                print('!!!Best Val Epoch: {}, Best Val Loss:{:.4f}'.format(best_val_epoch, best_val_loss))
                #torch.save(lvae, 'lvae%d.pt' % args.lamda)

                # test on val set
                open('%s/test_fea.txt' % args.save_path, 'w').close()
                open('%s/test_tar.txt' % args.save_path, 'w').close()
                open('%s/test_pre.txt' % args.save_path, 'w').close()
                open('%s/test_rec.txt' % args.save_path, 'w').close()


                for data_test, target_test in val_loader:
                    target_test_en = torch.Tensor(target_test.shape[0], args.num_classes)
                    target_test_en.zero_()
                    target_test_en.scatter_(1, target_test.view(-1, 1), 1)  # one-hot encoding
                    target_test_en = target_test_en.to(device)
                    if args.cuda:
                        data_test, target_test = data_test.cuda(), target_test.cuda()
                    with torch.no_grad():
                        data_test, target_test = Variable(data_test), Variable(target_test)

                    mu_test, output_test, de_test = lvae.test(data_test, target_test_en, args)
                    output_test = torch.exp(output_test)
                    prob_test = output_test.max(1)[0]  # get the value of the max probability
                    pre_test = output_test.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    rec_test = (de_test - data_test).pow(2).sum((3, 2, 1))
                    mu_test = torch.Tensor.cpu(mu_test).detach().numpy()
                    target_test = torch.Tensor.cpu(target_test).detach().numpy()
                    pre_test = torch.Tensor.cpu(pre_test).detach().numpy()
                    rec_test = torch.Tensor.cpu(rec_test).detach().numpy()

                    with open('%s/test_fea.txt' % args.save_path, 'ab') as f_test:
                        np.savetxt(f_test, mu_test, fmt='%f', delimiter=' ', newline='\r')
                        f_test.write(b'\n')
                    with open('%s/test_tar.txt' % args.save_path, 'ab') as t_test:
                        np.savetxt(t_test, target_test, fmt='%d', delimiter=' ', newline='\r')
                        t_test.write(b'\n')
                    with open('%s/test_pre.txt' % args.save_path, 'ab') as p_test:
                        np.savetxt(p_test, pre_test, fmt='%d', delimiter=' ', newline='\r')
                        p_test.write(b'\n')
                    with open('%s/test_rec.txt' % args.save_path, 'ab') as l_test:
                        np.savetxt(l_test, rec_test, fmt='%f', delimiter=' ', newline='\r')
                        l_test.write(b'\n')


                # test on test set
                for data_omn, target_omn in test_loader:
                    tar_omn = torch.from_numpy(args.num_classes * np.ones(target_omn.shape[0]))
                    if args.cuda:
                        data_omn = data_omn.cuda()
                    with torch.no_grad():
                        data_omn = Variable(data_omn)


                    mu_omn, output_omn, de_omn = lvae.test(data_omn, target_test_en, args)
                    output_omn = torch.exp(output_omn)
                    prob_omn = output_omn.max(1)[0]  # get the value of the max probability
                    pre_omn = output_omn.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    rec_omn = (de_omn - data_omn).pow(2).sum((3, 2, 1))
                    mu_omn = torch.Tensor.cpu(mu_omn).detach().numpy()
                    tar_omn = torch.Tensor.cpu(tar_omn).detach().numpy()
                    pre_omn = torch.Tensor.cpu(pre_omn).detach().numpy()
                    rec_omn = torch.Tensor.cpu(rec_omn).detach().numpy()

                    with open('%s/test_fea.txt' % args.save_path, 'ab') as f_test:
                        np.savetxt(f_test, mu_omn, fmt='%f', delimiter=' ', newline='\r')
                        f_test.write(b'\n')
                    with open('%s/test_tar.txt' % args.save_path, 'ab') as t_test:
                        np.savetxt(t_test, tar_omn, fmt='%d', delimiter=' ', newline='\r')
                        t_test.write(b'\n')
                    with open('%s/test_pre.txt' % args.save_path, 'ab') as p_test:
                        np.savetxt(p_test, pre_omn, fmt='%d', delimiter=' ', newline='\r')
                        p_test.write(b'\n')
                    with open('%s/test_rec.txt' % args.save_path, 'ab') as l_test:
                        np.savetxt(l_test, rec_omn, fmt='%f', delimiter=' ', newline='\r')
                        l_test.write(b'\n')


    open('%s/train_fea.txt' % args.save_path, 'w').close()  # clear
    np.savetxt('%s/train_fea.txt' % args.save_path, train_fea, delimiter=' ', fmt='%f')
    open('%s/train_tar.txt' % args.save_path, 'w').close()
    np.savetxt('%s/train_tar.txt' % args.save_path, train_tar, delimiter=' ', fmt='%d')
    open('%s/train_rec.txt' % args.save_path, 'w').close()
    np.savetxt('%s/train_rec.txt' % args.save_path, train_rec, delimiter=' ', fmt='%f')

    fea_omn = np.loadtxt('%s/test_fea.txt' % args.save_path)
    tar_omn = np.loadtxt('%s/test_tar.txt' % args.save_path)
    pre_omn = np.loadtxt('%s/test_pre.txt' % args.save_path)
    rec_omn = np.loadtxt('%s/test_rec.txt' % args.save_path)

    open('%s/test_fea.txt' % args.save_path, 'w').close()  # clear
    np.savetxt('%s/test_fea.txt' % args.save_path, fea_omn, delimiter=' ', fmt='%f')
    open('%s/test_tar.txt' % args.save_path, 'w').close()
    np.savetxt('%s/test_tar.txt' % args.save_path, tar_omn, delimiter=' ', fmt='%d')
    open('%s/test_pre.txt' % args.save_path, 'w').close()
    np.savetxt('%s/test_pre.txt' % args.save_path, pre_omn, delimiter=' ', fmt='%d')
    open('%s/test_rec.txt' % args.save_path, 'w').close()
    np.savetxt('%s/test_rec.txt' % args.save_path, rec_omn, delimiter=' ', fmt='%d')

    return best_val_loss, best_val_epoch


if __name__ == '__main__':

    args = get_args()
    control_seed(args)

    if args.dataset == "MNIST":
        load_dataset = MNIST_Dataset()
        args.num_classes = 6
        in_channel = 1
    elif args.dataset == "CIFAR10":
        load_dataset = CIFAR10_Dataset()
        args.num_classes = 6
        in_channel = 3
    elif args.dataset == "SVHN":
        load_dataset = SVHN_Dataset()
        args.num_classes = 6
        in_channel = 3
    elif args.dataset == "CIFARAdd10":
        load_dataset = CIFARAdd10_Dataset()
        args.num_classes = 4
        in_channel = 3
    elif args.dataset == "CIFARAdd50":
        load_dataset = CIFARAdd50_Dataset()
        args.num_classes = 4
        in_channel = 3
    elif args.dataset == "TinyImageNet":
        load_dataset = TinyImageNet_Dataset()
        args.num_classes = 20
        in_channel = 3
    elif args.dataset == "CIFAR100":
        load_dataset = CIFAR100_Dataset()
        args.num_classes = 15
        in_channel = 3
    elif args.dataset == "CIFARAddN":
        load_dataset = CIFARAddN_Dataset()
        args.num_classes = 4
        in_channel = 3

    exp_name = utils.get_exp_name(args)
    print("Experiment: %s" % exp_name)

    ### run experiment 1/5 times
    for run_idx in range(args.exp, args.exp+1):
        print("Begin to Run Exp %s..." %run_idx)
        args.run_idx = run_idx
        seed_sampler = int(args.seed_sampler.split(' ')[run_idx])
        # seed_sampler = None
        save_path = 'results/%s/%s/%s' %(args.dataset, exp_name, str(run_idx))
        args.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        latent_dim = 32
        if args.encode_z:
            latent_dim += args.encode_z

        lvae = LVAE(in_ch=in_channel,
                    out_ch64=64, out_ch128=128, out_ch256=256, out_ch512=512,
                    kernel1=1, kernel2=2, kernel3=3, padding0=0, padding1=1, stride1=1, stride2=2,
                    flat_dim32=32, flat_dim16=16, flat_dim8=8, flat_dim4=4, flat_dim2=2, flat_dim1=1,
                    latent_dim512=512, latent_dim256=256, latent_dim128=128, latent_dim64=64, latent_dim32=latent_dim,
                    num_class=args.num_classes, dataset=args.dataset, args=args)

        use_cuda = torch.cuda.is_available() and True
        device = torch.device("cuda" if use_cuda else "cpu")

        # data loader
        train_dataset, val_dataset, test_dataset = load_dataset.sampler(seed_sampler, args)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        # Model
        lvae.cuda()
        nllloss = nn.NLLLoss().to(device)

        # optimzer
        optimizer = optim.SGD(lvae.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
        print('decreasing_lr: ' + str(decreasing_lr))
        beta = DeterministicWarmup(n=50, t_max=1)  # Linear warm-up from 0 to 1 over 50 epoch
        if args.beta_anneal != 0:
            args.beta_anneal = DeterministicWarmup(n=args.beta_anneal, t_max=args.beta_z)


        if args.eval:
            # load train model
            states = torch.load(os.path.join(args.save_path, 'model.pkl'))
            lvae.load_state_dict(states['model'])

            ocr_test(args, lvae, train_loader, val_loader, test_loader)

        else:
            # Prepare summary writer
            if args.tensorboard:
                log_dir = "runs/%s" % (exp_name)
                writer = SummaryWriter(log_dir)

            best_val_loss, best_val_epoch = train(args, lvae)
            print('Finally!Best Epoch: {},  Best Val Loss: {:.4f}'.format(best_val_epoch, best_val_loss))

            if args.use_model:
                # load train model
                states = torch.load(os.path.join(args.save_path, 'model.pkl'))
                lvae.load_state_dict(states['model'])

            # perform test
            ocr_test(args, lvae, train_loader, val_loader, test_loader)
