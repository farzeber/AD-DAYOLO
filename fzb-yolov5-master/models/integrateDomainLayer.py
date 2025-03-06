'''
@Project ：yolov5-master 
@File    ：fzbLayers.py
@IDE     ：PyCharm 
@Author  ：付卓彬
@Date    ：2023/7/10
'''
import functools
from typing import Any, Optional, Tuple
from torch.autograd import Function
import torch.nn as nn
import torch
from models.common import  Conv
import torch.nn.functional as F
from utils.loss import compute_integdomain_loss

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

class advGRL(nn.Module):
    def     __init__(self):
        super(advGRL, self).__init__()

    def forward(self, *input, daLoss):
        self.daLoss = daLoss
        if self.daLoss < -1.:
            raise Exception
        if self.daLoss > 0.3:  # 拒绝反向传播，只训练分类器
            return GradientReverseFunction.apply(*input, 0.)
        else:
            weight =   (1 - self.daLoss) /   self.daLoss
            return GradientReverseFunction.apply(*input,weight)



# 域classifier
class _ImageDA(nn.Module):
    def   __init__(self,dim):
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


#域自适应网络   集中domain
class IntegDANLayer(nn.Module):

    def __init__(self, inputc):
        super(IntegDANLayer, self).__init__()
        self.kernel = 4
        self.stride = 4
        self.advGRL = advGRL()
        # self.F1 = nn.Sequential(GRL(),Conv(inputc[0],128,3,1), Conv(128,256,3,2), Conv(256,64,3,1))
        # self.F2 = nn.Sequential(GRL(),Conv(inputc[1],256,3,1), Conv(256,64,3,1))
        # self.F3 = nn.Sequential(GRL(),Conv(inputc[2],512,3,1), Conv(512,128,3,1), Conv(128,32,3,1))
        self.F1 = nn.Sequential(Conv(inputc[0],128,3,1), Conv(128,256,3,2), Conv(256,64,3,1))
        self.F2 = nn.Sequential(Conv(inputc[1],256,3,1), Conv(256,64,3,1))
        self.F3 = nn.Sequential(Conv(inputc[2],512,3,1), Conv(512,128,3,1), Conv(128,32,3,1))

        self.F12 = nn.Sequential(Conv(128,32,3,1), Conv(32, 64, 3, 2))

        self.F123 = nn.Sequential(Conv(96,16, 3, 1), Conv(16, 1, 3, 1), _ImageDA(1))

    def forward_ori(self, input,domainlables = 1):
        feature1, feature2, feature3 = input
        feature1 = self.F1(feature1)
        feature2 = self.F2(feature2)
        feature12 = self.F12(torch.cat([feature1, feature2], dim=1))
        feature3 = self.F3(feature3)
        output = self.F123(torch.cat([feature12, feature3], dim=1))
        return output


    def forward(self, input, domainlables = 1):
        if not isinstance(domainlables, list):
            return

        # 计算loss，根据loss来编排网络结构
        domainLabelsTensor = torch.tensor(domainlables)
        feature1, feature2, feature3 = input

        feature1_test = self.F1(feature1.detach())
        feature2_test = self.F2(feature2.detach())
        feature12_test = self.F12(torch.cat([feature1_test,feature2_test], dim=1))
        feature3_test = self.F3(feature3.detach())
        output = self.F123(torch.cat([feature12_test,feature3_test], dim=1))
        daloss = compute_integdomain_loss(output, domainLabelsTensor, feature1.device)  # 改改los

        # 开始真正反向传播
        feature1 = self.advGRL(feature1,daLoss=daloss.item())
        feature2 = self.advGRL(feature2,daLoss=daloss.item())
        feature3 = self.advGRL(feature3,daLoss=daloss.item())
        feature1 = self.F1(feature1)
        feature2 = self.F2(feature2)
        feature12 = self.F12(torch.cat([feature1, feature2], dim=1))
        feature3 = self.F3(feature3)
        output1 = self.F123(torch.cat([feature12, feature3], dim=1))

        return output1

    def forwardxxxx(self, input):
        feature1, feature2, feature3 = input

        feature1_path = extract_image_patches(feature1, ksizes=[self.kernel, self.kernel],
                                      strides=[self.stride,self.stride],
                                      rates=[1, 1],
                                      padding='same')

        feature2_path = extract_image_patches(feature2, ksizes=[self.kernel, self.kernel],
                                      strides=[self.stride,self.stride],
                                      rates=[1, 1],
                                      padding='same')

        feature3_path = extract_image_patches(feature3, ksizes=[self.kernel, self.kernel],
                                      strides=[self.stride,self.stride],
                                      rates=[1, 1],
                                      padding='same')

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