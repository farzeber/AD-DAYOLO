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
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression_tensor, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from torchvision.ops import RoIAlign
from utils.torch_utils import de_parallel

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

    def __init__(self, model,device):
        super(InstanceLayer, self).__init__()
        m = de_parallel(model).model[-1]  # Detect() module
        mm = de_parallel(model).model[-2]
        self.nc = m.nc  # number of classes
        self.no = m.nc + 5  # number of outputs per anchor
        self.outfag = True
        self.anchors = m.anchors
        self.nl = m.nl  # number of detection layers
        self.na = m.na # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.ROI = RoIAlign(output_size =7,spatial_scale = 1.,sampling_ratio = -1)
        ch = mm.ch
        self.modelLayer = nn.ModuleList()
        for ci in ch:
            sequence = [GRL(),nn.Linear(ci*7*7,1024),nn.ReLU(),nn.Dropout(p=0.5),nn.Linear(1024,1024),nn.ReLU(),nn.Dropout(p=0.5),nn.Linear(1024,1)]
            self.modelLayer.append(nn.Sequential(*sequence))
        self.instance_loss = nn.BCEWithLogitsLoss()
        self.device =device

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

    def forward(self, features,x, domainLabels):
        torch.cuda.empty_cache()
        print("1:{}MB".format(torch.cuda.memory_allocated(0)/1E6))
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            # 得到实例  features, x,imagesShape
            # features = featuresAndDetect[0:-1]
            # x = featuresAndDetect[-1]
            z = []  # inference output
            preds = []
            ROILayer = []
            print("2:{}MB".format(torch.cuda.memory_allocated(0) / 1E6))
            for i in range(self.nl):              # 循环每一个layer 其实可以去掉
                bs, _, ny, nx,_ = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
                print("3:{}MB{}".format(torch.cuda.memory_allocated(0) / 1E6,i))
                grid, anchor_grid = self._make_grid(nx, ny, i)
                print("4:{}MB{}".format(torch.cuda.memory_allocated(0) / 1E6,i))
                y = x[i].sigmoid()
                print("4.1:{}MB{}".format(torch.cuda.memory_allocated(0) / 1E6,i))
                xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                print("4.2:{}MB{}".format(torch.cuda.memory_allocated(0) / 1E6,i))
                xy = (xy * 2 + grid) # xy
                wh = (wh * 2) ** 2 * anchor_grid  # wh
                print("4.3:{}MB{}".format(torch.cuda.memory_allocated(0) / 1E6,i))
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

                # 非极大值印制  不同的layer list of detections, on (n,6) tensor per image [xyxy, conf, cls]
                pred = non_max_suppression_tensor(z[i], max_det=4)    # list 对应图片batch [n,5]  (xyxy, conf, cls) list of detections, on (n,6) tensor per image [xyxy, conf, cls]
                preds.append(pred)
                detlist = []
                print("5:{}MB".format(torch.cuda.memory_allocated(0) / 1E6))
                for j, det in enumerate(pred):  # per image
                    # det[:, :4] = scale_coords(imagesShape.shape[2:], det[:, :4], features[i].shape).round()  # 重新调整坐标 其实不需要调整
                    wholeImageDet = torch.tensor([0,0,ny,nx],device=self.device).view(-1,4)   # nms后可能没有框选中
                    roiVector = self.ROI(features[i][j:j+1], [torch.cat((det[...,:4],wholeImageDet),dim=0)])
                    detlist.append(roiVector)
                # 截取特征图  得到roi向量
                # feature[i] [batchsize,3,w,h]   detlist [batchsize,n,6]  # 提取Roi长度
                ROILayer.append(detlist)
                print("6:{}MB".format(torch.cuda.memory_allocated(0) / 1E6))
                del grid,anchor_grid

            print("7:{}MB".format(torch.cuda.memory_allocated(0) / 1E6))
            instanceLoss = 0
            imageNums = 0
            for seq,value in enumerate(ROILayer):
                for batch,featureSeq in enumerate(value):
                    if featureSeq.shape[0] !=0:
                        featureSeq = featureSeq.view(featureSeq.shape[0],-1)
                        predInstance = self.modelLayer[seq](featureSeq)
                        instanceLabel = torch.ones_like(predInstance)*domainLabels[batch]
                        instanceLoss += self.instance_loss(predInstance,instanceLabel)
                        imageNums +=1

             # 然后进行翻转 全连接层


            del ROILayer,preds,z
            # output = self.model(input)
            print("8:{}MB".format(torch.cuda.memory_allocated(0) / 1E6))
            return instanceLoss/(imageNums if imageNums!=0 else 1)

class InterToOut(nn.Module):
    def __init__(self,ch):
        super(InterToOut, self).__init__()
        self.ch = ch

    def forward(self, input):
        return input


