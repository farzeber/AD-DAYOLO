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
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression_tensor, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from torchvision.ops import RoIAlign
from utils.torch_utils import de_parallel
from models.common import  Conv

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
        self.modelLayer = nn.ModuleList()
        self.convLayer = nn.ModuleList()
        for ci in ch:
            sequence = [nn.Linear(4116,100),nn.ReLU(),nn.Dropout(p=0.5),nn.Linear(100,100),nn.ReLU(),nn.Dropout(p=0.5),nn.Linear(100,1)]
            self.convLayer.append(nn.Sequential(GRL(),nn.Conv2d(ci,84,1)))
            self.modelLayer.append(nn.Sequential(*sequence))
        self.instance_loss = nn.BCEWithLogitsLoss()

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid(y, x, indexing='ij')
        else:
            yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid

    def forward(self, features,x):
        z = []  # inference output
        preds = []
        ROILayer = []
        for i in range(self.nl):              # 循环每一个layer 其实可以去掉
            bs, _, ny, nx,_ = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            grid, anchor_grid = self._make_grid(nx, ny, i)
            grid = grid.to(x[0].device)
            anchor_grid = anchor_grid.to(x[0].device)
            y = x[i].sigmoid()
            xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
            xy = (xy * 2 + grid) # xy
            wh = (wh * 2) ** 2 * anchor_grid  # wh
            y = torch.cat((xy, wh, conf), 4)
            z.append(y.view(bs, -1, self.no))

            # 非极大值印制  不同的layer list of detections, on (n,6) tensor per image [xyxy, conf, cls]
            pred = non_max_suppression_tensor(z[i], max_det=25, iou_thres = 0.7)    # list 对应图片batch [n,5]  (xyxy, conf, cls) list of detections, on (n,6) tensor per image [xyxy, conf, cls]
            preds.append(pred)
            detlist = []

            for j, det in enumerate(pred):  # per image
                # det[:, :4] = scale_coords(imagesShape.shape[2:], det[:, :4], features[i].shape).round()  # 重新调整坐标 其实不需要调整
                roiVector = self.ROI(features[i][j:j+1], [det[...,:4]])
                detlist.append(roiVector)
            # 截取特征图  得到roi向量
            # feature[i] [batchsize,3,w,h]   detlist [batchsize,n,6]  # 提取Roi长度
            ROILayer.append(detlist)
            del grid,anchor_grid

        out = torch.zeros(self.nl, x[0].shape[0], 85, 1, device=x[0].device)
        for seq,value in enumerate(ROILayer):
            for batch,featureSeq in enumerate(value):
                if featureSeq.shape[0] !=0:
                    #featureSeq = featureSeq.view(featureSeq.shape[0],-1)
                    predInstance = self.convLayer[seq](featureSeq)
                    predInstance = predInstance.view(predInstance.shape[0],-1)
                    predInstance = self.modelLayer[seq](predInstance)
                    out[seq,batch,:featureSeq.shape[0]] = predInstance   #这里表示每张图片的多个roi向量

        del ROILayer,preds,z
        # output = self.model(input)
        return out

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

class DDAYOLOImageDA(nn.Module):
    def __init__(self, cis):
        super(DDAYOLOImageDA,self).__init__()
        self.grl = GRL()
        self.model1 = _ImageDA(cis[0])
        self.model2 = _ImageDA(cis[1])
        self.model3 = _ImageDA(cis[2])

    def forward(self, features):

        features1x2, features1x4, features1x8 = features
        features1x2 = self.grl(features1x2)
        features1x4 = self.grl(features1x4)
        features1x8 = self.grl(features1x8)
        output = self.model1(features1x2), self.model2(features1x4), self.model3(features1x8)
        return output

class DDAYOLOMultiLabelClassfier(nn.Module):
    def __init__(self,dim, classSum = 10):
        super(DDAYOLOMultiLabelClassfier,self).__init__()
        self.dim=dim  # feat layer          256*H*W for vgg16
        self.avl = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.conv_lst = nn.Conv2d(self.dim, classSum, 1, 1, 0)

    def forward(self,x):

        x = self.avl(x)
        x = self.conv_lst(x)
        output = x.squeeze(-1).squeeze(-1)
        #output = torch.sigmoid(output)  # （batchsize，2，H，W)
        return output


#域自适应网络
class DDAInstanceDC(nn.Module):

    def __init__(self, dims):
        self.dims = dims
        super(DDAInstanceDC, self).__init__()
        self.avl = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.m = nn.ModuleList(nn.Conv2d(x, 1, 1, 1, 0) for x in dims)

    def forward(self, x):
        output = []
        for i,feature in enumerate(x):
            x = self.avl(feature)
            x = self.m[i](x)
            output.append(x.squeeze(-1).squeeze(-1))
            #output = torch.sigmoid(output)  # （batchsize，2，H，W)
        return output