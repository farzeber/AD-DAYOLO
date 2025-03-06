'''
@Project ：yolov5-master
@File    ：fzbLayers.py
@IDE     ：PyCharm
@Author  ：付卓彬
@Date    ：2023/7/9 17:37
'''
import functools
from typing import Any, Optional, Tuple
from torch.autograd import Function
import torch.nn as nn
import torch
from torchvision.ops import RoIAlign


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


def pack_hook(x):
    #print("Packing", x.grad_fn)
    return x

def unpack_hook(x):
    #print("Unpacking", x.grad_fn)
    return x

#实例对齐网络
class InstanceLayer(nn.Module):

    def __init__(self, detectLayer, ch):
        super(InstanceLayer, self).__init__()
        m = detectLayer
        self.nc = m.nc  # number of classes
        self.no = m.nc + 5  # number of outputs per anchor
        self.anchors = m.anchors
        self.nl = m.nl  # number of detection layers
        self.na = m.na # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.ROI = RoIAlign(output_size =7,spatial_scale = 1.,sampling_ratio = -1)
        self.ROIL = RoIAlign(output_size =7,spatial_scale = 1.,sampling_ratio = -1)
        self.ROIM = RoIAlign(output_size =5,spatial_scale = 1.,sampling_ratio = -1)
        self.ROIH = RoIAlign(output_size =3,spatial_scale = 1.,sampling_ratio = -1)
        self.ROIList = torch.nn.ModuleList([self.ROIL, self.ROIM, self.ROIH])

        self.modelLayer = nn.ModuleList()
        self.convLayer = nn.ModuleList()
        for ci in ch:
            sequence = [nn.Linear(4116,100),nn.ReLU(),nn.Dropout(p=0.5),nn.Linear(100,100),nn.ReLU(),nn.Dropout(p=0.5),nn.Linear(100,1)]
            self.convLayer.append(nn.Sequential(GRL(),nn.Conv2d(ci,84,1)))
            self.modelLayer.append(nn.Sequential(*sequence))
        self.instance_loss = nn.BCEWithLogitsLoss()


    def forward(self, features,x):

        # output = self.model(input)
        return features

class InterToOut(nn.Module):
    def __init__(self,ch):
        super(InterToOut, self).__init__()
        self.ch = ch

    def forward(self, input):
        return input


