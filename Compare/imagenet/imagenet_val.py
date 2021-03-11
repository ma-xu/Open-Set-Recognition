import argparse
import os
import shutil
import time
import traceback
import warnings

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import sys
sys.path.append("../..")
import backbones.ImageNet as models
from Utils import adjust_learning_rate, progress_bar, Logger, mkdir_p, Evaluation
from BuildNet import BuildNet
import numpy as np

# Ignoring warnings
warnings.filterwarnings('ignore')

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--data', default='/home/g1007540910/DATA/ImageNet2012_O/', type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='old_resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=500, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--fp16', action='store_true',
                    help='Run model fp16 mode.')
parser.add_argument('--dali_cpu', action='store_true',
                    help='Runs CPU based version of DALI pipeline.')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                    '--static-loss-scale.')
parser.add_argument('--prof', dest='prof', action='store_true',
                    help='Only run 10 iterations for profiling.')
parser.add_argument('-t', '--test', action='store_true',
                    help='Launch test mode with preset arguments')

parser.add_argument("--local_rank", default=0, type=int)
#################################
parser.add_argument('--train_class_num', default=500, type=int, help='Classes used in training')
parser.add_argument('-v', '--val', default='val', type=str)


cudnn.benchmark = True

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, random_shuffle=True)
        #let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[0.8, 1.25],
                                                 random_area=[0.1, 1.0],
                                                 num_attempts=100)
        self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]

class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, random_shuffle=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]

best_prec1 = 0
args = parser.parse_args()

# checkpoint
if args.checkpoint is None:
    if args.fp16:
        args.checkpoint='checkpoints/imagenet/'+args.arch+'_FP16'
    else:
        args.checkpoint = 'checkpoints/imagenet/' + args.arch + '_FP32'


args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

# make apex optional
if args.fp16 or args.distributed:
    print("Import APEX!")
    try:
        from apex.parallel import DistributedDataParallel as DDP
        from apex.fp16_utils import *
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def main():
    global best_prec1, args

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    args.total_batch_size = args.world_size * args.batch_size

    if not os.path.isdir(args.checkpoint) and args.local_rank == 0:
        mkdir_p(args.checkpoint)

    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    if args.static_loss_scale != 1.0:
        if not args.fp16:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = BuildNet(backbone=args.arch, num_classes=args.train_class_num)


    model = model.cuda()
    if args.fp16:
        model = network_to_half(model)
    if args.distributed:
        # shared param/delay all reduce turns off bucketing in DDP, for lower latency runs this can improve perf
        # for the older version of APEX please use shared_param, for newer one it is delay_allreduce
        model = DDP(model, delay_allreduce=True)

    # define loss function (criterion) and optimizer
    # equals to psoftmax if input is ["normweight_fea2cen"]
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.static_loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   verbose=False)

    # optionally resume from a checkpoint
    title = 'ImageNet-' + args.arch
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        if args.local_rank == 0:
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
            logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Valid Top5.'])

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, args.val)


    crop_size = 224
    val_size = 256

    # pipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=traindir, crop=crop_size, dali_cpu=args.dali_cpu)
    # pipe.build()
    # train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

    pipe = HybridValPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=valdir, crop=crop_size, size=val_size)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))
    validate(val_loader, model)



def validate(val_loader, model,intervals=20):
    # switch to evaluate mode
    model.eval()
    if args.local_rank == 0:
        print("start evaluating...")

    normfea_list = []  # extracted feature norm
    energy_list = []  # energy value
    normweight_fea2cen_list = []
    Target_list = []
    Predict_list = []


    for i, data in enumerate(val_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda().long()
        val_loader_len = int(val_loader._size / args.batch_size)

        if args.local_rank == 0 and i%200 ==0:
            print(f"evaluating {i}\t/{val_loader_len}...")

        target = target.cuda(non_blocking=True)
        input_var = Variable(input)
        target_var = Variable(target)

        # compute output
        with torch.no_grad():
            out = model(input_var)

            normfea_list.append(out["norm_fea"])
            energy_list.append(out["energy"])
            normweight_fea2cen_list.append(out["normweight_fea2cen"])
            Target_list.append(target)
            _, predicted = (out['normweight_fea2cen']).max(1)
            Predict_list.append(predicted)

    normfea_list = torch.cat(normfea_list, dim=0)
    energy_list = torch.cat(energy_list, dim=0)
    normweight_fea2cen_list = torch.cat(normweight_fea2cen_list, dim=0)
    Target_list = torch.cat(Target_list, dim=0)
    Predict_list = torch.cat(Predict_list, dim=0)

    best_F1_possibility = 0
    best_F1_norm = 0
    best_F1_energy = 0

    # for these unbounded metric, we explore more intervals by *5 to achieve a relatively fair comparison.
    expand_factor = 5
    Predict_list_possibility = Predict_list.clone().detach()
    Predict_list_norm = Predict_list.clone().detach()
    Predict_list_energy = Predict_list.clone().detach()

    # possibility
    openmetric_possibility = normweight_fea2cen_list
    openmetric_possibility, _ = torch.softmax(openmetric_possibility, dim=1).max(dim=1)
    for thres in np.linspace(0.0, 1.0, intervals):
        Predict_list_possibility[openmetric_possibility < thres] = args.train_class_num
        eval = Evaluation(Predict_list_possibility.cpu().numpy(), Target_list.cpu().numpy())
        if eval.f1_measure > best_F1_possibility:
            best_F1_possibility = eval.f1_measure

        # norm
    openmetric_norm = normfea_list.squeeze(dim=1)
    threshold_min_norm = openmetric_norm.min().item()
    threshold_max_norm = openmetric_norm.max().item()
    for thres in np.linspace(threshold_min_norm, threshold_max_norm, expand_factor * intervals):
        Predict_list_norm[openmetric_norm < thres] = args.train_class_num
        eval = Evaluation(Predict_list_norm.cpu().numpy(), Target_list.cpu().numpy())
        if eval.f1_measure > best_F1_norm:
            best_F1_norm = eval.f1_measure

    # energy
    openmetric_energy = energy_list
    threshold_min_energy = openmetric_energy.min().item()
    threshold_max_energy = openmetric_energy.max().item()
    for thres in np.linspace(threshold_min_energy, threshold_max_energy, expand_factor * intervals):
        Predict_list_energy[openmetric_energy < thres] = args.train_class_num
        eval = Evaluation(Predict_list_energy.cpu().numpy(), Target_list.cpu().numpy())
        if eval.f1_measure > best_F1_energy:
            best_F1_energy = eval.f1_measure

    if args.local_rank == 0:
        print(f"Best Possibility F1 is: {best_F1_possibility} | Norm F1 is :{best_F1_norm} | Energy F1 is: {best_F1_energy}")
    return {
        "best_F1_possibility": best_F1_possibility,
        "best_F1_norm": best_F1_norm,
        "best_F1_energy": best_F1_energy
    }



def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    # try:
    #     main()
    # except Exception as e:
    #     print(e)
    #     traceback.print_exc()
    #     os.system("sudo poweroff")
    # print("DONE, FINISHED!!!")
    # os.system("sudo poweroff")
    main()
