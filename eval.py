#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import os
import random
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from torchvision.datasets.samplers import RandomClipSampler, UniformClipSampler, DistributedSampler
from dataset.ucf101 import UCF101
import moco.loader
from torch.utils.tensorboard import SummaryWriter
from utils.train_utils import adjust_learning_rate, accuracy, save_checkpoint, sanity_check, AverageMeter, ProgressMeter
from moco.loader import Augment_GPU_ft
from backbone.i3d import I3D
from backbone.r2plus1d import r2plus1d_18


model_names = ['I3D','r2plus1d_18']
parser = argparse.ArgumentParser(description='PyTorch Linear Classification Video Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='r3d_18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--num_class', default=101, type=int,
                    help='number of class')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-cs', '--crop_size', default=112, type=int, metavar='N',
                    help='crop size for video clip (default: 112)')
parser.add_argument('-fpc', '--frame_per_clip', default=16, type=int, metavar='N',
                    help='number of frame per video clip (default: 16)')
parser.add_argument('-sbc', '--step_between_clips', default=1, type=int, metavar='N',
                    help='number of steps between video clips (default: 1)')
parser.add_argument('-cpv', '--clip_per_video', default=10, type=int, metavar='N',
                    help='number of frame per video clip (default: 10)')
parser.add_argument('--lr', '--learning_rate', default=30., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_decay', '--learning_rate_decay', default=0.1, type=float,
                    metavar='LRD', help='learning rate decay', dest='lr_decay')
parser.add_argument('--warmup', action='store_true',
                    help='use warm up lr schedule')
parser.add_argument('--wp_lr', '--warmup_learning_rate', default=0.0025, type=float,
                    metavar='WLR', help='initial warmup learning rate', dest='wp_lr')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('--dropout', default=0.7, type=float, metavar='M',
                    help='dropout rate')
parser.add_argument('-p', '--print_freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--log_dir', default='logs_lincls', type=str,
                    help='path to the tensorboard log directory')
parser.add_argument('--ckp_dir', default='checkpoints_lincls', type=str,
                    help='path to the linear classification model directory')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--finetune', action='store_true',
                    help='finetune backbone instead of freeze')
parser.add_argument('--freeze', action='store_true',
                    help='freeze backbone instead of finetune')
parser.add_argument('--beta',default=0.5, type=float,
                    help='portion of the foreground')
best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    # args.pretrained = os.path.join(args.log_dir, "checkpoint_0199.pth.tar")
    if args.num_class == 101:
        if args.freeze:
            args.ckp_dir = os.path.join(args.log_dir, "linear", "checkpoints_UCF")
            args.log_dir = os.path.join(args.log_dir, "linear", "logs_UCF")
        if args.finetune:
            args.ckp_dir = os.path.join(args.log_dir, "finetune", "checkpoints_UCF200")
            args.log_dir = os.path.join(args.log_dir, "finetune", "logs_UCF200")
        args.dropout = 0.7

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == "r2plus1d_18":
        model = r2plus1d_18(drop=args.finetune, dropout=args.dropout)
        if args.freeze:
            model = r2plus1d_18(drop=False)
            for name, param in model.named_parameters():
                if name not in ['fc.weight', 'fc.bias']:
                    param.requires_grad = False
        # freeze all layers but the last fc
        # init the fc layer
        model.fc = nn.Linear(512, args.num_class, bias=True)
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()
    elif args.arch == 'I3D':
        model = I3D(num_classes=args.num_class, dropout_prob=args.dropout, with_classifier=True)
    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            print("missing", msg.missing_keys)
            # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model)#.cuda() for debug on cpu
    print(args)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    # assert len(parameters) == 2  # fc.weight, fc.bias
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            # if args.gpu is not None:
            #     # best_acc1 may be from a checkpoint from a different GPU
            #     best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    normalize_video = transforms_video.NormalizeVideo(mean=[0.43216, 0.394666, 0.37645],
                                                      std=[0.22803, 0.22145, 0.216989])


    aug_gpu = Augment_GPU_ft(args)
    
    video_augmentation_train = transforms.Compose(
        [
            transforms_video.ToTensorVideo(),
            transforms_video.RandomResizedCropVideo(args.crop_size, (0.2, 1)),
            transforms_video.RandomHorizontalFlipVideo(),
        ]
    )
    video_augmentation_val = transforms.Compose(
        [
            transforms_video.ToTensorVideo(),
            transforms_video.CenterCropVideo(args.crop_size),
            normalize_video,
        ]
    )
    data_dir = os.path.join(args.data, 'data')
    anno_dir = os.path.join(args.data, 'anno')
    train_dataset = UCF101(
        data_dir,
        anno_dir,
        args.frame_per_clip,
        args.step_between_clips,
        fold=1,
        train=True,
        transform=video_augmentation_train,
        num_workers=16
    )

    val_dataset = UCF101(
        data_dir,
        anno_dir,
        args.frame_per_clip,
        args.step_between_clips,
        fold=1,
        train=False,
        transform=video_augmentation_val,
        num_workers=16
    )
    train_sampler = RandomClipSampler(train_dataset.video_clips, 1)

    if args.distributed:
        train_sampler = DistributedSampler(train_sampler)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,
        multiprocessing_context="fork")

    # # Do not use DistributedSampler since it will destroy the testing iteration process
    val_sampler = UniformClipSampler(val_dataset.video_clips, args.clip_per_video)


    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.clip_per_video, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler,
        multiprocessing_context="fork")

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    if args.multiprocessing_distributed and args.gpu == 0:
        log_dir = "{}_bs={}_lr={}_cs={}_wd={}".format(args.log_dir, args.batch_size, args.lr, args.crop_size, args.weight_decay)
        writer = SummaryWriter(log_dir)
    else:
        writer = None
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, writer, aug_gpu)

        # evaluate on validation set
        if epoch == args.epochs-1 or epoch % 10 == 0:
            val_loss, acc1, acc5 = validate(val_loader, model, criterion, args)
            if writer is not None:
                writer.add_scalar('lincls_val/loss', val_loss, epoch)
                writer.add_scalar('lincls_val/acc1', acc1, epoch)
                writer.add_scalar('lincls_val/acc5', acc5, epoch)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            ckp_dir = "{}_bs={}_lr={}_cs={}_wd={}".format(args.ckp_dir, args.batch_size, args.lr,
                                                             args.crop_size,
                                                             args.weight_decay)
            save_checkpoint(epoch, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, ckp_dir, max_save=5, is_best=is_best)
            # if epoch == args.start_epoch:
            #     sanity_check(model.state_dict(), args.pretrained)
    print(args)

def train(train_loader, model, criterion, optimizer, epoch, args, writer, aug_gpu):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    # model.eval()
    if args.freeze:
        model.eval()
    if args.finetune: 
        model.train()

    end = time.time()
    for i, (video, target) in enumerate(train_loader):
        # measure data loading time

        data_time.update(time.time() - end)

        if args.gpu is not None:
            video = video.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        if args.finetune:
            video = aug_gpu(video)
        # compute output
        output = model(video)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), video.size(0))
        top1.update(acc1[0], video.size(0))
        top5.update(acc5[0], video.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            if writer is not None:
                total_iter = i + epoch * len(train_loader)
                writer.add_scalar('lincls_train/loss', loss, total_iter)
                writer.add_scalar('lincls_train/acc1', acc1, total_iter)
                writer.add_scalar('lincls_train/acc5', acc5, total_iter)
                writer.add_scalar('lincls_train_avg/lr', optimizer.param_groups[0]['lr'], total_iter)
                writer.add_scalar('lincls_train_avg/loss', losses.avg, total_iter)
                writer.add_scalar('lincls_train_avg/acc1', top1.avg, total_iter)
                writer.add_scalar('lincls_train_avg/acc5', top5.avg, total_iter)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (video, target) in enumerate(val_loader):
            if args.gpu is not None:
                video = video.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(video)
            output = output.view(-1, args.clip_per_video, args.num_class)
            target = target.view(-1, args.clip_per_video)
            output = torch.mean(output, dim=1)
            # make sure 10 clips belong to the same video
            for j in range(1, args.clip_per_video):
                assert all(target[:, 0]==target[:, j])
            target = target[:, 0]
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), video.size(0))
            top1.update(acc1[0], video.size(0))
            top5.update(acc5[0], video.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    main()
