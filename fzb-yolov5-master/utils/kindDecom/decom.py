'''
@Project ：kindFuxian 
@File    ：decom.py
@IDE     ：PyCharm 
@Author  ：付卓彬
@Date    ：2023/3/12 16:43 
'''

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, activeRelu = False):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        pad = k // 2
        #self.conv = nn.Conv2d(c1,c2,k, padding='same')
        self.conv = nn.Conv2d(c1, c2, k, padding= pad)
        self.relu = nn.ReLU()
        self.activeRelu = activeRelu

    def forward(self, x):
        out = self.conv(x)
        if self.activeRelu:
            out = self.relu(out)
        return out

class Upsame(nn.Module):
    def __init__(self, inchannel, outChannel):
        super().__init__()
        self.deconv_filter = nn.ConvTranspose2d(inchannel,outChannel,2,stride=2)

    def forward(self, x1, x2):
        out = self.deconv_filter(x1) #反卷积第一个
        #out = torch.concatenate((out,x2),axis=1)
        out = torch.cat((out, x2), axis=1)
        return out

class DecomLayer(nn.Module):

    def __init__(self,inchannel):
        super().__init__()
        self.decom = nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True), nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True), Conv(32,64,3,True), nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
                                   , Conv(64,128,3,True))
        self.conv1 = Conv(inchannel,32,3,True)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        self.conv2 = Conv(32,64,3,True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv3 = Conv(64,128,3,True)
        self.up8 = Upsame(128, 64)       #有contact 输出变两倍
        self.conv8 = Conv(128,64,3,True)
        self.up9 = Upsame(64, 32)         #有contact 输出变两倍
        self.conv9 = Conv(64,32,3,True)
        self.conv10 = Conv(32,3,1,False)
        self.l_conv2 = Conv(32,32,3,True)
        self.l_conv4 = Conv(64,1,1,False)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        up8 = self.up8(conv3,conv2)
        conv8 = self.conv8(up8)
        up9 = self.up9(conv8,conv1)
        conv9 = self.conv9(up9)
        conv10 = self.conv10(conv9)
        R_out = torch.sigmoid(conv10)

        l_conv2 = self.l_conv2(conv1)
        #l_conv3 = torch.concatenate((l_conv2, conv9), axis=1)
        l_conv3 = torch.cat((l_conv2, conv9), axis=1)
        l_conv4 = self.l_conv4(l_conv3)
        L_out = torch.sigmoid(l_conv4)
        return R_out, L_out
