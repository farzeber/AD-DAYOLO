'''
@Project ：yolov5-master 
@File    ：fzbLayers.py
@IDE     ：PyCharm 
@Author  ：付卓彬
@Date    ：2023/6/3 17:37 
'''
import functools
from typing import Any, Optional, Tuple
from torch.autograd import Function
import torch.nn as nn
import torch
from models.common import  Conv
from utils.hrDecom.deyolo import DENet
from models.zeroDceLayer import zeroDceEnhLayer

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



# 域检测layer
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=1, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=2, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
#此处网络结构
"""[Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), LeakyReLU(negative_slope=0.2, inplace=True), 
Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False), LeakyReLU(negative_slope=0.2, inplace=True),
 Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False), LeakyReLU(negative_slope=0.2, inplace=True),
 Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)), InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False), LeakyReLU(negative_slope=0.2, inplace=True),
 Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1))]"""

# 域classifier
class _ImageDA(nn.Module):
    def __init__(self,dim):
        super(_ImageDA,self).__init__()
        self.dim=dim  # feat layer          256*H*W for vgg16
        self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1,bias=False)
        self.Conv2=nn.Conv2d(512,2,kernel_size=1,stride=1,bias=False)
        self.reLu=nn.ReLU(inplace=False)

    def forward(self,x):
        x=self.reLu(self.Conv1(x))
        x=self.Conv2(x)
        #x = F.softmax(x, dim=1)  # （batchsize，2，H，W)
        return x

#域自适应网络
class DANLayer(nn.Module):

    def __init__(self, inputc,hiddenc = 128):
        super(DANLayer, self).__init__()
        self.outfag = True
        sequence = []
        sequence += [GRL(),Conv(inputc,hiddenc), Conv(hiddenc,1),_ImageDA(1)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        output = self.model(input)
        return output

class DEDALayer(nn.Module):
    def __init__(self):
        super(DEDALayer, self).__init__()
        # 炮哥的增强 1.直接使用 2. 更改一部分
        #self.model = DENet()
        self.model = zeroDceEnhLayer()

    def forward(self, feature, domianLabels):
        # 只对目标域标签进行特征恢复
        output = self.model(feature)
        # if domianLabels == 1:
        #     #output = feature
        #     output = self.model(feature)
        # else:
        #     domianLabels = torch.tensor(domianLabels)
        #     output = torch.zeros_like(feature, device=feature.device)
        #     output[domianLabels == 1] = self.model(feature[domianLabels == 1], torch.mean(feature[domianLabels == 0]).detach())
        #     #output[domianLabels == 1] = self.model(feature[domianLabels == 1])
        #     output[domianLabels == 0] = feature[domianLabels == 0]
        # # # 原域的特征图不参与改变
        # # if domianLabels != 1:   # eval 评估
        # #     output[domianLabels == 0] = feature[domianLabels == 0]
        return output
