'''
@Project ：yolov5-master 
@File    ：priorENlayer.py
@IDE     ：PyCharm 
@Author  ：付卓彬
@Date    ：2023/9/10 14:20 
'''
import functools
from typing import Any, Optional, Tuple
from torch.autograd import Function
import torch.nn as nn
import torch
from torch.nn import functional as F
from models.common import Conv

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


class CIN(nn.Module):
    def __init__(self,chin):
        super().__init__()
        self.chout = chin // 2
        self.me = nn.Sequential(nn.Linear(chin,chin // 2), nn.Linear(chin//2, chin//2))
        self.st = nn.Sequential(nn.Linear(chin, chin // 2), nn.Linear(chin // 2, chin // 2))

    def forward(self, feature, lagm):
        assert feature.shape[1] == self.chout
        me = self.me(lagm)
        st = self.st(lagm)
        shape = feature.shape
        feature1 = feature.view(shape[0], shape[1], -1)
        featureMe = torch.mean(feature1, dim=2,keepdim=True)
        featureMenorm = featureMe.view(shape[0], shape[1], 1, 1)
        meNorm = me.view(shape[0], shape[1], 1,1)
        featureSt = torch.var(feature1, dim=2, keepdim=True)
        featureSt = torch.sqrt(featureSt + 1e-6)
        featureStnorm = featureSt.view(shape[0], shape[1], 1, 1)
        stNorm = st.view(shape[0], shape[1], 1, 1)
        output = (feature - featureMenorm) / featureStnorm
        output = stNorm * output + meNorm
        return  output

# 带有先验知识的对抗学习
class FCM(nn.Module):
    def __init__(self, chin, seq):
        super().__init__()
        self.seq = seq
        self.conv1 = nn.Conv2d(chin, chin//2, 1,1,0)
        self.cin = CIN(chin)
        self.conv2 = nn.Conv2d(chin//2, chin,1,1,0)

    def forward(self, feature):
        feature, lagm = feature
        lagm = lagm[self.seq]
        feature1 = self.conv1(feature)
        feature2 = self.cin(feature1, lagm)
        feature3 = self.conv2(feature2)
        output = feature + feature3
        return output

class LAGM(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(Conv(3,8,3,1,act = nn.LeakyReLU()),Conv(8,16,5,2,act = nn.LeakyReLU()),
                                  Conv(16,32,3,1,act = nn.LeakyReLU()),Conv(32,64,5,2,act = nn.LeakyReLU()),Conv(64,128,3,1,act = nn.LeakyReLU()),nn.AdaptiveAvgPool2d((1,1)))
        # self.model = nn.Sequential(Conv(3, 8, 3, 1), Conv(8, 16, 5, 2),
        #                            Conv(16, 32, 3, 1), Conv(32, 64, 5, 2),
        #                            Conv(64, 128, 3, 1), nn.AdaptiveAvgPool2d(1, 1))

        self.fc = nn.Sequential(nn.Linear(128,128), nn.Linear(128, 96))

    def forward(self, feature):
        feature1 = self.model(feature)
        shape = feature.shape
        feature2 = feature1.view(shape[0], -1)
        output = self.fc(feature2)
        output = torch.split(output,[32, 64], dim=1)
        return output

class OutEqualInput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,feature):
        return feature