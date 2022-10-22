import torch
import torch.nn.functional as F
from torch.utils import data
# import torch.optim as optim
import os
import sys
import argparse
import time
import csv
import numpy as np
from dataset.ucf101 import UCF101
from r2plus1d import r2plus1d_18
import moco.loader
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from torchvision.datasets.samplers import RandomClipSampler, UniformClipSampler

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='finetune', type=str)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--wd', default=1e-3, type=float)
parser.add_argument('--img_dim', default=112, type=int)
parser.add_argument('--test_dim', default=112, type=int)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--channel', default=128, type=int)
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--num_class', default=101, type=int)
parser.add_argument('--ft', default=10, type=float)
parser.add_argument('-cpv', '--clip_per_video', default=10, type=int, metavar='N',
                    help='number of frame per video clip (default: 10)')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--log_dir', default='logs_moco', type=str,
                    help='path to the tensorboard log directory')
args = parser.parse_args()

def test(model, dataloader, mode):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for idx, (data, _, label) in enumerate(dataloader):
            data = data.to(device)
            label = label.to(device)
            # end = time.time()
            feat = model(data.squeeze(0))
            feat = feat.mean(0)
            features.append(feat.detach().cpu())
            labels.append(label[0].cpu())
            if idx % 100 == 0:
                print('already down %d in %s mode'%(idx,mode))
                print(label)
    # bar.finish()
    features = torch.stack(features)
    labels = torch.stack(labels)

    torch.save(features, args.log_dir + '/%s_feat.pth.tar'%mode)
    torch.save(labels, args.log_dir + '/%s_label.pth.tar'%mode)
    return features, labels

def retrieve(key, query, kl, ql):
    print(query.shape, key.shape, ql.shape, kl.shape)
    ql = ql.reshape(-1, 1)
    kl = kl.reshape(-1, 1)
    # query = F.normalize(query, dim=1, p=2)
    # key = F.normalize(key, dim=1, p=2)
    query = query - query.mean(dim=0, keepdim=True)
    key = key - key.mean(dim=0, keepdim=True)
    query = F.normalize(query, dim=1, p=2)
    key = F.normalize(key, dim=1, p=2)
    sim = torch.matmul(query, key.transpose(0, 1))
    for k in [1, 5, 10, 20, 50]:
        topkval, topkidx = torch.topk(sim, k, dim=1)
        acc = torch.any(kl[:, 0][topkidx]==ql, dim=1).float().mean().item()
        print(acc)

def main():
    global device; device = torch.device('cuda')
    # Data loading code
    normalize_video = transforms_video.NormalizeVideo(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])



    video_augmentation = transforms.Compose(
        [
            transforms_video.ToTensorVideo(),
            transforms_video.CenterCropVideo(args.test_dim),
            normalize_video,
        ]
    )
    data_dir = os.path.join(args.data, 'data')
    anno_dir = os.path.join(args.data, 'anno')
    audio_augmentation = moco.loader.DummyAudioTransform()
    # train_augmentation = {'video': video_augmentation_train, 'audio': audio_augmentation}
    augmentation = {'video': video_augmentation, 'audio': audio_augmentation}

    train_dataset = UCF101(
        data_dir,
        anno_dir,
        16,
        1,
        fold=1,
        train=True,
        transform=augmentation,
        num_workers=16
    )
    train_sampler = UniformClipSampler(train_dataset.video_clips, args.clip_per_video)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.clip_per_video, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,
        multiprocessing_context="fork")

    val_dataset = UCF101(
        data_dir,
        anno_dir,
        16,
        1,
        fold=1,
        train=False,
        transform=augmentation,
        num_workers=16
    )
    # Do not use DistributedSampler since it will destroy the testing iteration process
    val_sampler = UniformClipSampler(val_dataset.video_clips, args.clip_per_video)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.clip_per_video, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler,
        multiprocessing_context="fork")
    from i3d import Normalize
    #model = models.__dict__[args.arch]()
    model = r2plus1d_18()
    model.fc = Normalize(2)
    

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

    model = model.to(device)
    trainfeat, trainlabel = test(model, train_loader, 'train')
    testfeat, testlabel = test(model, val_loader, 'test')
    retrieve(trainfeat, testfeat, trainlabel, testlabel)

if __name__ == '__main__':
    main()
