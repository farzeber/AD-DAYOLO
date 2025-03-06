'''
@Project ：yolov5-master 
@File    ：integQKVidaLayer.py
@IDE     ：PyCharm 
@Author  ：付卓彬
@Date    ：2024/1/26 14:55 
'''
import functools
import math
from typing import Any, Optional, Tuple
from torch.autograd import Function
import torch.nn as nn
import torch
from models.common import  Conv
import torch.nn.functional as F
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


class IntegQKVDANLayerOri(nn.Module):

    def __init__(self, inputc):
        super(IntegQKVDANLayerOri, self).__init__()

        self.F1 = nn.Sequential(GRL(),Conv(inputc[0],128,3,1), Conv(128,32,3,1), Conv(32,32,3,2), Conv(32,32,3,2))
        self.F2 = nn.Sequential(GRL(),Conv(inputc[1],256,3,1), Conv(256,64,3,1),Conv(64,32,3,1),Conv(32,32,3,2))
        self.F3 = nn.Sequential(GRL(),Conv(inputc[2],512,3,1), Conv(512,256,3,1), Conv(256,128,3,1),Conv(128,32,3,1))

        # self.F12 = nn.Sequential(Conv(128,32,3,1), Conv(32, 64, 3, 2))
        # self.F123 = nn.Sequential(Conv(96,16, 3, 1), Conv(16, 1, 3, 1), _ImageDA(1))
        # self.h 控制多少个通道为一个
        self.h = 1
        # 这里面有 全连接层 ，只能用于训练中
        self.ssa = SimplifiedScaledDotProductAttention(d_model=400, h=self.h)

        self.F123 = nn.Sequential(Conv(32, 16, 3, 1), Conv(16, 1, 3, 1), _ImageDA(1))

    def forward(self, input):
        feature1, feature2, feature3 = input
        # f1，f2,f3 逐梯度下降
        feature1 = self.F1(feature1)
        feature2 = self.F2(feature2)
        feature3 = self.F3(feature3)

        # f1，f2使用QKV融合  通道自注意力机制   维度变化
        Fqk = feature1 + feature2
        shape = Fqk.shape
        Fqk = Fqk.view(Fqk.size(0), shape[1] // self.h,-1)
        if self.training:
            Fqk = self.ssa(Fqk,Fqk,Fqk)
        outvalue = Fqk.view(shape[0], shape[1], shape[2], shape[3])
        output = outvalue + feature3
        output = self.F123(output)
        return output

class IntegQKVDANLayer(nn.Module):

    def __init__(self, inputc):
        super(IntegQKVDANLayer, self).__init__()

        self.F1 = nn.Sequential(GRL(),Conv(inputc[0],128,3,1), Conv(128,32,3,1), Conv(32,32,3,2), Conv(32,32,3,2))
        self.F2 = nn.Sequential(GRL(),Conv(inputc[1],256,3,1), Conv(256,64,3,1),Conv(64,32,3,1),Conv(32,32,3,2))
        self.F3 = nn.Sequential(GRL(),Conv(inputc[2],512,3,1), Conv(512,256,3,1), Conv(256,128,3,1),Conv(128,32,3,1))

        self.convQ = Conv(96,96,1,1)
        self.convK = Conv(96,96,1,1)
        self.convV = Conv(96,96,1,1)

        self.F123 = nn.Sequential(Conv(96, 32, 3, 1), Conv(32, 1, 3, 1), _ImageDA(1))

    def forward(self, input):
        feature1, feature2, feature3 = input
        # f1，f2,f3 逐梯度下降
        feature1 = self.F1(feature1)
        feature2 = self.F2(feature2)
        feature3 = self.F3(feature3)

        # f1，f2使用QKV融合  通道自注意力机制   维度变化
        # Fqk = feature1 + feature2
        # shape = Fqk.shape
        # Fqk = Fqk.view(Fqk.size(0), shape[1] // self.h,-1)
        # if self.training:
        #     Fqk = self.ssa(Fqk,Fqk,Fqk)
        # outvalue = Fqk.view(shape[0], shape[1], shape[2], shape[3])
        # output = outvalue + feature3
        # output = self.F123(output)
        # 引入空间自注意力机制
        feature123 = torch.cat((feature1,feature2,feature3), dim=1)
        shape = feature123.shape

        featureQ = self.convQ(feature123)
        featureK = self.convK(feature123)
        featureV = self.convV(feature123)

        Q = featureQ.view(shape[0],shape[1], -1).permute(0, 2, 1)
        K = featureK.view(shape[0],shape[1], -1).permute(0, 1, 2)
        V = featureV.view(shape[0], shape[1], -1).permute(0, 2, 1)

        att = torch.matmul(Q, K) / torch.sqrt(torch.tensor(shape[1]))  #
        att = torch.softmax(att, -1)
        #att = self.dropout(att)

        tranfOut = torch.matmul(att, V).permute(0, 2, 1).contiguous().view(shape[0], shape[1], shape[2], shape[3])
        scOut = tranfOut + feature123
        output = self.F123(scOut)
        return output

class HrIntegQKVDANLayer(nn.Module):

    def __init__(self, inputc):
        super(HrIntegQKVDANLayer, self).__init__()

        self.F1 = nn.Sequential(GRL(),Conv(inputc[0],128,3,1), Conv(128,32,3,1), Conv(32,32,3,2), Conv(32,32,3,2))
        self.F2 = nn.Sequential(GRL(),Conv(inputc[1],256,3,1), Conv(256,64,3,1),Conv(64,32,3,1),Conv(32,32,3,2))
        self.F3 = nn.Sequential(GRL(),Conv(inputc[2],512,3,1), Conv(512,128,3,1), Conv(128,32,3,1))

        # self.F12 = nn.Sequential(Conv(128,32,3,1), Conv(32, 64, 3, 2))
        # self.F123 = nn.Sequential(Conv(96,16, 3, 1), Conv(16, 1, 3, 1), _ImageDA(1))
        # self.h 控制多少个通道为一个
        self.h = 1
        # 这里面有 全连接层 ，只能用于训练中
        self.ssa = SimplifiedScaledDotProductAttention(d_model=400, h=self.h)
        self.F123 = nn.Sequential(Conv(32, 16, 3, 1), Conv(16, 1, 3, 1), _ImageDA(1))

        # self.hrLayerF1 = nn.Sequential(*[Conv(inputc[0],128), Conv(128,3), nn.Conv2d(3, 3, kernel_size=1, stride=1,bias=False), nn.Sigmoid()])
        # self.hrLayerF2 = nn.Sequential(*[Conv(inputc[1],128), Conv(128,3), nn.Conv2d(3, 3, kernel_size=1, stride=1,bias=False), nn.Sigmoid()])
        # self.hrLayerF3 = nn.Sequential(*[Conv(inputc[2],128), Conv(128,3), nn.Conv2d(3, 3, kernel_size=1, stride=1,bias=False), nn.Sigmoid()])
        self.hrLayerF1 = nn.Sequential(*[nn.Conv2d(inputc[0], 3, kernel_size=3, stride=1,bias=False,padding=1), nn.Sigmoid()])
        self.hrLayerF2 = nn.Sequential(*[nn.Conv2d(inputc[1], 3, kernel_size=3, stride=1,bias=False,padding=1), nn.Sigmoid()])
        self.hrLayerF3 = nn.Sequential(*[nn.Conv2d(inputc[2], 3, kernel_size=3, stride=1,bias=False,padding=1), nn.Sigmoid()])


    def forward(self, input):
        feature1c, feature2c, feature3c = input
        # f1，f2,f3 逐梯度下降
        feature1 = self.F1(feature1c)
        feature2 = self.F2(feature2c)
        feature3 = self.F3(feature3c)

        # f1，f2使用QKV融合  通道自注意力机制   维度变化
        Fqk = feature1 + feature2
        shape = Fqk.shape
        Fqk = Fqk.view(Fqk.size(0), shape[1] // self.h,-1)
        if self.training:
            Fqk = self.ssa(Fqk,Fqk,Fqk)
        outvalue = Fqk.view(shape[0], shape[1], shape[2], shape[3])
        output = outvalue + feature3
        output = self.F123(output)

        hr1 = self.hrLayerF1(feature1c)
        hr2 = self.hrLayerF2(feature2c)
        hr3 = self.hrLayerF3(feature3c)
        return output, [hr1,hr2,hr3]

class PriorIntegQKVDANLayer(nn.Module):

    def __init__(self, inputc):
        super(PriorIntegQKVDANLayer, self).__init__()

        self.F1 = nn.Sequential(GRL(),Conv(inputc[0],128,3,1), Conv(128,32,3,1), Conv(32,32,3,2), Conv(32,32,3,2))
        self.F2 = nn.Sequential(GRL(),Conv(inputc[1],256,3,1), Conv(256,64,3,1),Conv(64,32,3,1),Conv(32,32,3,2))
        self.F3 = nn.Sequential(GRL(),Conv(inputc[2],512,3,1), Conv(512,128,3,1), Conv(128,32,3,1))

        # self.F12 = nn.Sequential(Conv(128,32,3,1), Conv(32, 64, 3, 2))
        # self.F123 = nn.Sequential(Conv(96,16, 3, 1), Conv(16, 1, 3, 1), _ImageDA(1))
        # self.h 控制多少个通道为一个
        self.h = 1
        # 这里面有 全连接层 ，只能用于训练中
        self.ssa = SimplifiedScaledDotProductAttention(d_model=400, h=self.h)

        self.F123 = nn.Sequential(Conv(32, 16, 3, 1), Conv(16, 1, 3, 1), nn.Conv2d(1,1,3,1,1), nn.Sigmoid())

    def forward(self, input):
        feature1, feature2, feature3 = input
        # f1，f2,f3 逐梯度下降
        feature1 = self.F1(feature1)
        feature2 = self.F2(feature2)
        feature3 = self.F3(feature3)

        # f1，f2使用QKV融合  通道自注意力机制   维度变化
        Fqk = feature1 + feature2
        shape = Fqk.shape
        Fqk = Fqk.view(Fqk.size(0), shape[1] // self.h,-1)
        if self.training:
            Fqk = self.ssa(Fqk,Fqk,Fqk)
        outvalue = Fqk.view(shape[0], shape[1], shape[2], shape[3])
        output = outvalue + feature3
        output = self.F123(output)
        return output

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images

# for CSWAlayer
def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks

class CSWALayer(nn.Module):

    def __init__(self, inputc):
        super(CSWALayer, self).__init__()
        self.kernel = 4
        self.stride = 4
        # self.F1 = nn.Sequential(GRL(),Conv(inputc[0],128,3,1), Conv(128,32,3,1))
        # self.F2 = nn.Sequential(GRL(),Conv(inputc[1],256,3,1), Conv(256,64,3,1),Conv(64,32,3,1))
        # self.F3 = nn.Sequential(GRL(),Conv(inputc[2],512,3,1), Conv(512,128,3,1), Conv(128,32,3,1))
        self.F1 = nn.Sequential(GRL(),Conv(inputc[0],128,3,1))
        self.F2 = nn.Sequential(GRL(),Conv(inputc[1],256,3,1), Conv(256,128,3,1))
        self.F3 = nn.Sequential(GRL(),Conv(inputc[2],512,3,1), Conv(512,256,3,1), Conv(256,128,3,1))
        # self.F12 = nn.Sequential(Conv(128,32,3,1), Conv(32, 64, 3, 2))
        # self.F123 = nn.Sequential(Conv(96,16, 3, 1), Conv(16, 1, 3, 1), _ImageDA(1))
        # self.h 控制多少个通道为一个

        # self.F123 = nn.Sequential(Conv(32,16, 3, 1), Conv(16, 1, 3, 1), _ImageDA(1))
        self.F123 = nn.Sequential(_ImageDA(128))


    def forward(self, input):

        feature1, feature2, feature3 = input
        if feature1.shape[2] != 80 or feature1.shape[3] != 80:
            return
        feature11 = self.F1(feature1)
        feature22 = self.F2(feature2)
        feature33 = self.F3(feature3)
        feature1_path = extract_image_patches(feature11, ksizes=[self.kernel, self.kernel],
                                      strides=[self.stride,self.stride],
                                      rates=[1, 1],
                                      padding='same')

        feature2_path = extract_image_patches(feature22, ksizes=[self.kernel, self.kernel],
                                      strides=[self.stride,self.stride],
                                      rates=[1, 1],
                                      padding='same')

        feature3_path = extract_image_patches(feature33, ksizes=[self.kernel, self.kernel],
                                      strides=[self.stride,self.stride],
                                      rates=[1, 1],
                                      padding='same')

        # # f1，f2,f3 逐梯度下降
        # feature1 = self.F1(feature1)
        # feature2 = self.F2(feature2)
        # feature3 = self.F3(feature3)

        # f1，f2使用QKV融合  通道自注意力机制   维度变化
        Q = feature3_path.permute(0, 2, 1)
        K = feature2_path.permute(0, 1, 2)
        V = feature1_path.permute(0, 2, 1)

        att = torch.matmul(Q, K) / torch.sqrt(torch.tensor(Q.shape[2]))  #

        #跨规模匹配
        att_shape = att.shape
        att = att.view(att_shape[0],att_shape[1],int(math.sqrt(att_shape[2])),int(math.sqrt(att_shape[2])))
        att = att.repeat(1,1,2,2)
        att = att.view(att_shape[0],att_shape[1],-1)
        att = torch.softmax(att, -1)
        # att = self.dropout(att)

        tranfOut = torch.matmul(att, V).permute(0, 2, 1).contiguous().view(feature33.shape[0], feature33.shape[1], feature33.shape[2], feature33.shape[3])

        output = self.F123(tranfOut)
        # output11 = self.F11(feature11)
        # output22 = self.F11(feature22)
        # output33 = self.F11(feature33)
        return output

