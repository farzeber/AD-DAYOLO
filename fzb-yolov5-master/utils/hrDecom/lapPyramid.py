'''
@Project ：yolov5-master 
@File    ：lapPyramid.py
@IDE     ：PyCharm 
@Author  ：付卓彬
@Date    ：2023/10/23 15:05 
'''
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re
import torch
import torch.nn as nn
from torch.nn import functional as F


class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3, kernel_size=5, channels=3,device='cpu'):
        super().__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel(kernel_size, channels,device)
    def gauss_kernel(self, kernel_size, channels, device):
        kernel = cv2.getGaussianKernel(kernel_size, 0).dot(
            cv2.getGaussianKernel(kernel_size, 0).T)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).repeat(
            channels, 1, 1, 1)
        kernel = torch.nn.Parameter(data=kernel, requires_grad=False).to(device)
        return kernel

    def conv_gauss(self, x, kernel):
        n_channels, _, kw, kh = kernel.shape
        x = torch.nn.functional.pad(x, (kw // 2, kh // 2, kw // 2, kh // 2),
                                    mode='reflect')  # replicate    # reflect
        x = torch.nn.functional.conv2d(x, kernel, groups=n_channels)
        return x

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def pyramid_down(self, x):
        return self.downsample(self.conv_gauss(x, self.kernel))

    def upsample(self, x):
        up = torch.zeros((x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2),
                         device=x.device)
        up[:, :, ::2, ::2] = x * 4

        return self.conv_gauss(up, self.kernel)

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            down = self.pyramid_down(current)
            up = self.upsample(down)
            shape = current.shape

            up = up[:, :, :shape[2], :shape[3]]
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[0]
        for level in pyr[1:]:
            up = self.upsample(image)
            image = up + level
        return image
