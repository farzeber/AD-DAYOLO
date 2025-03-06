'''
@Project ：yolov5-master 
@File    ：featEnHancerLayer.py
@IDE     ：PyCharm 
@Author  ：付卓彬
@Date    ：2023/8/21 17:00 
'''
import functools
from typing import Any, Optional, Tuple
from torch.autograd import Function
import torch.nn as nn
import torch
from torch.nn import functional as F
from models.common import  Conv
from fightingcv_attention.attention.MobileViTv2Attention import *
from fightingcv_attention.attention.SimplifiedSelfAttention import SimplifiedScaledDotProductAttention

class GradientReverseFunction(Function):
    """
    重写自定义的梯度计算方式
    """
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GRL(nn.Module):
    def __init__(self):
        super(GRL, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

class FENlayer(nn.Module):
    def __init__(self, ch, co = 32):
        super(FENlayer,self).__init__()
        self.singleCovLayer = nn.ModuleList()
        for ci in ch:
            self.singleCovLayer.append(nn.Conv2d(ci,co,3,1,1))
        self.Conv1 = nn.Sequential(nn.Conv2d(co,co,3,1,1), nn.LeakyReLU())
        self.Conv2 = nn.Sequential(nn.Conv2d(co,co,3,1,1), nn.LeakyReLU())
        self.Conv3 = nn.Sequential(nn.Conv2d(co,co,3,1,1), nn.LeakyReLU())
        self.Conv4 = nn.Sequential(nn.Conv2d(co,co,3,1,1), nn.LeakyReLU())

    def forward(self, input):
        features = []
        for seq,feature in enumerate(input):
            features.append(self.singleCovLayer[seq](feature))
        outfeatures = []
        for feature in features:
            interm1 = self.Conv1(feature)
            interm2 = self.Conv2(interm1)
            interm3 = self.Conv3(interm2 + interm1)
            interm4 = self.Conv4(interm3)
            outfeatures.append(interm4 + feature)
        return outfeatures

class SAFA(nn.Module):
    def __init__(self, ch):
        super(SAFA, self).__init__()
        self.FConv = nn.Sequential(nn.Conv2d(ch,ch,7,4,2),nn.LeakyReLU())
        self.FqConv = nn.Sequential( nn.Conv2d(ch,ch,3,2,1), nn.LeakyReLU())
        self.h = 4

    def forward(self,FF, Fq):
        F1 = self.FConv(FF)
        Fq1 = self.FqConv(Fq)
        Fqk = F1 + Fq1

        # 两特征图融合
        out=F.interpolate(Fqk ,scale_factor=8,mode="bilinear")
        return out


class FEHancer(nn.Module):
    def __init__(self,ch):
        super(FEHancer, self).__init__()
        self.fenlayer = FENlayer(ch,32)
        self.safa = SAFA(32)
        self.last = nn.Sequential(nn.Conv2d(32,3,3,1,1))     # 激活函数选什么，如果需要残差连接输入图片

    def forward(self, feature):
        oriPic = feature[-1]
        feature = feature[:-1]
        FF, Fq, Fo = self.fenlayer(feature)
        Fo = F.interpolate(Fo, scale_factor=8, mode="bilinear")
        safaF = self.safa(FF, Fq)
        out = self.last(safaF + Fo)
        #out = F.sigmoid(out)
        oriPic = F.interpolate(oriPic ,scale_factor=0.25,mode="bilinear")
        #out = out * oriPic
        return out    # 看看这个输出图 怎么样


class _ImageDA(nn.Module):
    def __init__(self,dim, classSum = 10):
        super(_ImageDA,self).__init__()
        self.dim=dim  # feat layer          256*H*W for vgg16
        self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1,bias=False)
        self.Conv2=nn.Conv2d(512,classSum,kernel_size=1,stride=1,bias=False)
        self.reLu=nn.ReLU(inplace=False)
        self.avl = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.nextlayer = nn.Linear(classSum, classSum)

    def forward(self,x):
        x=self.reLu(self.Conv1(x))
        x=self.Conv2(x)
        x = self.avl(x)
        output = x.view(x.shape[0], -1)
        output = self.nextlayer(output)
        #output = torch.sigmoid(output)  # （batchsize，2，H，W)
        return output

class FEHclassifier(nn.Module):

    def __init__(self, ch):
        super(FEHclassifier, self).__init__()
        self.modelS = nn.Sequential(Conv(ch, 3, 3, 1), _ImageDA(3))
        self.modelT = nn.Sequential(GRL(),Conv(ch,3, 3, 1), _ImageDA(3))

    def forward(self, input):
        outputS = self.modelS(input)     # 进行multilabel标签
        outputT = self.modelT(input)

        return outputS, outputT,input

class inputMapOut(nn.Module):
    def __init__(self):
        super(inputMapOut, self).__init__()

    def forward(self,input):
        return input



