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
from models.fzbLayers import DANLayer

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

# 带有先验知识的对抗学习
class PEN(nn.Module):
    def __init__(self, ci):
        super(PEN, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(ci,ci,1,1,0), nn.BatchNorm2d(ci),nn.ReLU(),
                                   nn.Conv2d(ci,ci,3,1,1), nn.BatchNorm2d(ci),nn.ReLU(),
                                   nn.Conv2d(ci,ci,1,1,0), nn.BatchNorm2d(ci),nn.ReLU(),
                                   nn.Conv2d(ci,1,3,1,1), nn.Tanh())

    def forward(self, feature):
        return self.model(feature)

# 用来对目标域图像进行更改, 后面可以考虑对这个模块进行更改
class RFRB(nn.Module):
    def __init__(self, ci , cout ):
        super(RFRB, self).__init__()
        self.model = nn.Sequential(nn.MaxPool2d(2,stride=2),
                                   nn.Conv2d(ci,cout,3,1,1),  nn.ReLU(),
                                   nn.Conv2d(cout,cout,3,1,1),  nn.ReLU(),
                                   nn.Conv2d(cout,cout,3,1,1),  nn.ReLU())

    def forward(self, feature, domianLabels):
        x1, x2 = feature
        # 只对目标域标签进行特征恢复

        output = x2 + self.model(x1)
        # 原域的特征图不参与改变
        if domianLabels != 1:   # 训练时不改正常光图像。 测试全部改
            output[domianLabels == 0] = x2[domianLabels == 0]
        return output


# 后面可能会考虑融合 三个图片级别的分层 仿照integDAyolo
class ImagePENs(nn.Module):
    def __init__(self, cis):
        super(ImagePENs,self).__init__()
        self.grl = GRL()
        self.model1 = PEN(cis[0])
        self.model2 = PEN(cis[1])
        self.model3 = PEN(cis[2])

    def forward(self, features):

        features1x2, features1x4, features1x8 = features
        features1x2 = self.grl(features1x2)
        features1x4 = self.grl(features1x4)
        features1x8 = self.grl(features1x8)
        output = self.model1(features1x2), self.model2(features1x4), self.model3(features1x8)
        return output

class IntegImagePENs(nn.Module):

    def __init__(self, inputc):
        super(IntegImagePENs, self).__init__()
        self.F1 = nn.Sequential(GRL(),Conv(inputc[0],inputc[0],1,1),Conv(inputc[0],128,3,1), Conv(128,256,3,2), Conv(256,64,3,1))
        self.F2 = nn.Sequential(GRL(),Conv(inputc[1],inputc[1],1,1),Conv(inputc[1],256,3,1), Conv(256,64,3,1))
        self.F3 = nn.Sequential(GRL(),Conv(inputc[2],inputc[2],1,1),Conv(inputc[2],512,3,1), Conv(512,128,3,1), Conv(128,32,3,1))

        self.F12 = nn.Sequential(Conv(128,32,3,1), Conv(32, 64, 3, 2))

        self.F123 = nn.Sequential(Conv(96,16, 3, 1),nn.Conv2d(16,1,3,1,1), nn.Tanh())


    def forward(self, input):
        feature1, feature2, feature3 = input
        feature1 = self.F1(feature1)
        feature2 = self.F2(feature2)
        feature12 = self.F12(torch.cat([feature1,feature2], dim=1))
        feature3 = self.F3(feature3)
        output = self.F123(torch.cat([feature12,feature3], dim=1))
        return output

#
class ImagePENsAndImageDoamin(nn.Module):
    def __init__(self, cis):
        super(ImagePENsAndImageDoamin,self).__init__()
        self.grl = GRL()
        self.model1 = PEN(cis[0])
        self.model2 = PEN(cis[1])
        self.model3 = DANLayer(cis[2])

    def forward(self, features):

        features1x2, features1x4, features1x8 = features
        features1x2 = self.grl(features1x2)
        features1x4 = self.grl(features1x4)
        #features1x8 = self.grl(features1x8)
        output = self.model1(features1x2), self.model2(features1x4), self.model3(features1x8)
        return output