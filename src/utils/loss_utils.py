#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import csv
import lpips

lpips_fn = lpips.LPIPS(net="alex").eval()


def l1_loss(network_output, gt, mean=True):
    loss = torch.abs((network_output - gt))
    if mean:
        return loss.mean()
    else:
        return loss


def l2_loss(network_output, gt, mean=True):
    loss = (network_output - gt) ** 2
    if mean:
        return loss.mean()
    else:
        return loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def psnr(inputs, targets, valid_mask=None, reduction="mean"):
    assert reduction in ["mean", "none"]
    value = (inputs - targets) ** 2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == "mean":
        return -10 * torch.log10(torch.mean(value))
    elif reduction == "none":
        return -10 * torch.log10(torch.mean(value, dim=tuple(range(value.ndim)[1:])))


def lpips_loss(pred, ref):
    global lpips_fn
    pred = pred[None].permute(0, 3, 1, 2)
    ref = ref.permute(0, 3, 1, 2)
    lpips_fn = lpips_fn.to(pred.device)
    dist = lpips_fn.forward(pred, ref)
    return dist.mean()


def write_csv(csv_path, row, include_header=False):
    with open(csv_path, "a") as csvfile:
        # Add a new row to the end of the file
        writer = csv.writer(csvfile, delimiter=",")
        if include_header:
            writer.writerow(
                [
                    "name",
                    "step",
                    "psnr",
                    "ssim",
                    "lpips",
                    "rendering_time",
                ]
            )
        writer.writerow(row)
        csvfile.close()
