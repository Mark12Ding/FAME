import random
from torch.utils.data import Sampler
from torchvision.datasets.video_utils import VideoClips
import torch
import torch.nn as nn
import kornia
import torchvision.transforms as transforms
import numpy as np
from kornia.augmentation.container import VideoSequential


class FAME(nn.Module):
    def __init__(self, crop_size=112, beta=0.5, device="cpu", eps=1e-8):
        super(FAME, self).__init__()
        self.crop_size = crop_size
        gauss_size = int(0.1 * crop_size) // 2 * 2 + 1
        self.gauss = kornia.filters.GaussianBlur2d(
            (gauss_size, gauss_size),
            (gauss_size / 3, gauss_size / 3))
        self.device = device
        self.eps = eps
        self.beta = beta # control the portion of foreground

    #### min-max normalization
    def norm_batch(self, matrix):
        # matrix : B*H*W
        B, H, W = matrix.shape
        matrix = matrix.flatten(start_dim=1)
        matrix -= matrix.min(dim=-1, keepdim=True)[0]
        matrix /= (matrix.max(dim=-1, keepdim=True)[0] + self.eps)
        return matrix.reshape(B, H, W)

    def batched_bincount(self, x, dim, max_value):
        target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
        values = torch.ones_like(x)
        target.scatter_add_(dim, x, values)
        return target
    
    def getSeg(self, mask, video_clips):
        # input mask:B, H, W; video_clips:B, C, T, H, W
        # return soft seg mask: B, H, W
        B, C, T, H, W = video_clips.shape
        video_clips_ = video_clips.mean(dim=2) # B, C, H, W
        img_hsv = kornia.color.rgb_to_hsv(video_clips_.reshape(-1, C, H, W))  # B, C, H, W
        sampled_fg_index = torch.topk(mask.reshape(B, -1), k=int(0.5 * H * W), dim=-1)[1]  # shape B * K
        sampled_bg_index = torch.topk(mask.reshape(B, -1), k=int(0.1 * H * W), dim=-1, largest=False)[1]  # shape B * K
        
        dimH, dimS, dimV = 10, 10, 10
        img_hsv = img_hsv.reshape(B, -1, H, W)  # B * C * H * W
        img_h = img_hsv[:, 0]
        img_s = img_hsv[:, 1]
        img_v = img_hsv[:, 2]
        hx = (img_s * torch.cos(img_h * 2 * np.pi) + 1) / 2
        hy = (img_s * torch.sin(img_h * 2 * np.pi) + 1) / 2
        h = torch.round(hx * (dimH - 1) + 1)
        s = torch.round(hy * (dimS - 1) + 1)
        v = torch.round(img_v * (dimV - 1) + 1)
        color_map = h + (s - 1) * dimH + (v - 1) * dimH * dimS  # B, H, W
        color_map = color_map.reshape(B, -1).long()
        col_fg = color_map.gather(index=sampled_fg_index, dim=-1)  # B * K
        col_bg = color_map.gather(index=sampled_bg_index, dim=-1)  # B * K
        dict_fg = self.batched_bincount(col_fg, dim=1, max_value=dimH * dimS * dimV)  # B * (dimH * dimS * dimV)
        dict_bg = self.batched_bincount(col_bg, dim=1, max_value=dimH * dimS * dimV)  # B * (dimH * dimS * dimV)
        dict_fg = dict_fg.float()
        dict_bg = dict_bg.float() + 1
        dict_fg /= (dict_fg.sum(dim=-1, keepdim=True) + self.eps)
        dict_bg /= (dict_bg.sum(dim=-1, keepdim=True) + self.eps)

        pr_fg = dict_fg.gather(dim=1, index=color_map)
        pr_bg = dict_bg.gather(dim=1, index=color_map)
        refine_mask = pr_fg / (pr_bg + pr_fg)

        mask = self.gauss(refine_mask.reshape(-1, 1, H, W))
        mask = self.norm_batch(mask.reshape(-1, H, W))
        num_fg = int(self.beta * H * W)
        sampled_index = torch.topk(mask.reshape(B, -1), k=num_fg, dim=-1)[1]
        mask = torch.zeros_like(mask).reshape(B, -1)
        b_index = torch.LongTensor([[i]*num_fg for i in range(B)])
        mask[b_index.view(-1), sampled_index.view(-1)] = 1
        return mask.reshape(B, H, W)

    def getmask(self, video_clips):
        # input video_clips: B, C, T, H, W
        # return soft seg mask: B, H, W
        B, C, T, H, W = video_clips.shape
        im_diff = (video_clips[:, :, 0:-1] - video_clips[:, :, 1:]).abs().sum(dim=1).mean(dim=1)  # B, H, W
        mask = self.gauss(im_diff.reshape(-1, 1, H, W))
        mask = self.norm_batch(mask.reshape(-1, H, W))  # B, H, W
        mask = self.getSeg(mask, video_clips)
        return mask

    def forward(self, video_clips):
        # return video_clips : B, C, T, H, W
        mask = self.getmask(video_clips)
        B, C, T, H, W = video_clips.shape
        index = torch.randperm(B, device=self.device)
        video_fuse = video_clips[index] * (1 - mask).reshape(-1, 1, 1, H, W) + video_clips * mask.reshape(-1, 1, 1, H, W)
        return video_fuse


def Augment_GPU_pre(args):
    crop_size = args.crop_size
    radius = int(0.1*crop_size)//2*2+1
    sigma = random.uniform(0.1, 2)
    # For k400 parameter:
    mean = torch.tensor([0.43216, 0.394666, 0.37645])
    std = torch.tensor([0.22803, 0.22145, 0.216989])

    normalize_video = kornia.augmentation.Normalize(mean, std)
    aug_list = VideoSequential(
        kornia.augmentation.RandomGrayscale(p=0.2),
        kornia.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
        # kornia.augmentation.RandomHorizontalFlip(),
        kornia.augmentation.RandomGaussianBlur((radius, radius), (sigma, sigma), p=0.5),
        normalize_video,
        data_format="BCTHW",
        same_on_frame=True)
    return aug_list


def Augment_GPU_ft(args):
    # For k400 parameter:
    mean = torch.tensor([0.43216, 0.394666, 0.37645])
    std = torch.tensor([0.22803, 0.22145, 0.216989])
    normalize_video = kornia.augmentation.Normalize(mean, std)
    aug_list = VideoSequential(
        kornia.augmentation.RandomGrayscale(p=0.2),
        kornia.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
        normalize_video,
        data_format="BCTHW",
        same_on_frame=True)
    return aug_list


class RandomTwoClipSampler(Sampler):
    """
    Samples two clips for each video randomly

    Arguments:
        video_clips (VideoClips): video clips to sample from
    """
    def __init__(self, video_clips):
        if not isinstance(video_clips, VideoClips):
            raise TypeError("Expected video_clips to be an instance of VideoClips, "
                            "got {}".format(type(video_clips)))
        self.video_clips = video_clips

    def __iter__(self):
        idxs = []
        s = 0
        # select two clips for each video, randomly
        for c in self.video_clips.clips:
            length = len(c)
            if length < 2:
                sampled = [s, s]
            else:
                sampled = torch.randperm(length)[:2] + s
                sampled = sampled.tolist()
            s += length
            idxs.append(sampled)
        # shuffle all clips randomly
        random.shuffle(idxs)
        return iter(idxs)

    def __len__(self):
        return len(self.video_clips.clips)


class DummyAudioTransform(object):
    """This is a dummy audio transform.

    It ignores actual audio data, and returns an empty tensor. It is useful when
    actual audio data is raw waveform and has a varying number of waveform samples
    which makes minibatch assembling impossible

    """

    def __init__(self):
        pass

    def __call__(self, _audio):
        return torch.zeros(0, 1, dtype=torch.float)
