'''
@Project ：yolov5-master 
@File    ：contrastDomainLayer.py
@IDE     ：PyCharm 
@Author  ：付卓彬
@Date    ：2023/7/22 12:44 
'''
import functools
from typing import Any, Optional, Tuple
from torch.autograd import Function
import torch.nn as nn
import torch
from models.common import  Conv
import torch.nn.functional as F

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

class contraDANLayer(nn.Module):
    def __init__(self, ch):
        super(contraDANLayer, self).__init__()
        layerList = [GRL(), Conv(ch,ch,3,2), Conv(ch,8,3,1), Conv(8,16,5,2), Conv(16,32,3,1), Conv(32,64,5,2), Conv(64,128,3,1),
                     nn.AdaptiveAvgPool2d(output_size=(1,1))]
        self.model = nn.Sequential(*layerList)
        self.nextlayer = nn.Linear(128, 128)

    def forward(self, input):
        # befeature, endfeature = input
        # befeature = befeature.detach()
        # feature = torch.cat((befeature, endfeature), dim=0)
        output = self.model(input)
        shape = output.shape
        output = output.view(shape[0],-1)
        output = self.nextlayer(output)
        return output

