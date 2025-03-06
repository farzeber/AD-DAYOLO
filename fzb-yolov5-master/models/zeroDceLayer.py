'''
@Project ：yolov5-master 
@File    ：zeroDceLayer.py
@IDE     ：PyCharm 
@Author  ：付卓彬
@Date    ：2023/10/31 16:50 
'''
import functools
from typing import Any, Optional, Tuple
from torch.autograd import Function
import torch.nn as nn
import torch


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


#####  这里借鉴一下zero-dce，函数，原来的想法是保持 像素通道颜色比例不变，但实际发现这样无法更改影响

# 用来对目标域图像进行更改, 后面可以考虑对这个模块进行更改
class zeroDceEnhLayer(nn.Module):
    def __init__(self, number_f = 32):
        super(zeroDceEnhLayer, self).__init__()

        self.cov1 = nn.Sequential(nn.Conv2d(3, number_f, 3,1,1), nn.ReLU(inplace=True))
        self.cov2 = nn.Sequential(nn.Conv2d(number_f, number_f, 3, 1, 1), nn.ReLU(inplace=True))
        self.cov3 = nn.Sequential(nn.Conv2d(number_f, number_f, 3, 1, 1), nn.ReLU(inplace=True))

        # 使用nn.tanh呢还是什么呢
        self.cov4 = nn.Sequential(nn.Conv2d(number_f*2, 3, 3, 1, 1), nn.Tanh())

        # self.mean_val = torch.nn.Parameter(data=torch.FloatTensor([0.6]), requires_grad=False)
        # self.pool = nn.AvgPool2d(16)
        # self.mlength = 16

    def forward(self, feature, dayPic_exp = None):
        # 得到a参数
        x_r = self.construct(feature)
        output = self.enhance(feature, x_r)

        ### 添加一个曝光亮度校正
        #output = self.regular_exp(x_r,output,dayPic_exp)

        return output

    def regular_exp(self,x_r,feature, dayPic_exp):
        if not (dayPic_exp is None):
          self.mean_val.data = (self.mean_val + dayPic_exp) / 2
        enhance_exp = torch.mean(feature)
        output = feature
        exp_dlta = self.mean_val - enhance_exp
        if exp_dlta > 0:
            a_mean = exp_dlta / (enhance_exp *(1 - enhance_exp) + 1e-6)
            x_r_mean = torch.mean(x_r)

            x_r_mean_adv = self.pool(x_r)
            shape = x_r_mean_adv.shape
            x_r_mean = x_r_mean_adv.view(shape[0], shape[1],shape[2],1,shape[3],1)
            x_r_mean = x_r_mean.repeat(1,1,1,self.mlength ,1,self.mlength )
            x_r_mean = x_r_mean.view(shape[0],shape[1],shape[2]*self.mlength ,shape[3]*self.mlength)

            x_r = x_r + torch.clamp( a_mean - x_r_mean, 0 , 1)            # 这里是全局调整，后面可以让小熊改成 局部曝光调整， 同时x_r = torch.clamp(x_r,0,1)
            x_r = torch.clamp(x_r,0, 1)
            # x = x + r1*(torch.pow(x,2)-x)
            output = feature + x_r*(feature - torch.pow(feature,2))
        return output

    def construct(self, x):
        """
         inference
         """
        x1 = self.cov1(x)
        x2 = self.cov2(x1)
        x3 = self.cov3(x2)
        x4 = self.cov4(torch.concatenate([x3,x1], dim=1))
        return x4


    ##########   迭代次数应该与图片原光照图像相关   8 次是对应低光图像
    ####
    ###
    def enhance(self, x, x_r):
        """
        enhance with the x_r
        """

        x = x*torch.exp(x_r*(x - torch.pow(x, 2) ))
        x = x*torch.exp(x_r*(x - torch.pow(x, 2) ))
        x = x*torch.exp(x_r*(x - torch.pow(x, 2) ))
        enhance_image_1 = x*torch.exp(x_r*(x - torch.pow(x, 2) ))
        x = enhance_image_1*torch.exp(x_r*(enhance_image_1 - torch.pow(enhance_image_1, 2) ))
        x = x*torch.exp(x_r*(x - torch.pow(x, 2) ))
        x = x*torch.exp(x_r*(x - torch.pow(x, 2) ))
        enhance_image = x*torch.exp(x_r*(x - torch.pow(x, 2) ))
        return enhance_image