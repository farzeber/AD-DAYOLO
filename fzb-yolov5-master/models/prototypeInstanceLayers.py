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
                           increment_path, xywh2xyxy, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from torchvision.ops import RoIAlign
from utils.torch_utils import de_parallel
import torchvision

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


# 源域实例模型
class protoSInst(nn.Module):
    def __init__(self,nc = 80,anchors = (),inCh =()):
        super().__init__()
        self.nc = nc
        self.feaProtypeList = torch.nn.ParameterList()
        for ch in inCh:
            fea = torch.nn.ParameterList([torch.nn.Parameter(data=torch.zeros([ch],dtype=torch.float), requires_grad=False) for j in range(nc)])
            self.feaProtypeList.append(fea)

        self.nl = len(anchors)  # number of detection layers
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)

    def forward(self,feature,target):
        if not torch.is_tensor(target):
            return
        with torch.no_grad():
            tcls, _, indices, _ = self.build_targets(feature,target)


            # 每个layer有类的原型
            for i,fea in enumerate(feature):
                b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx


                n = b.shape[0]  # number of targets
                if n:
                    # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                    # 这里从pred挑选 选择 与 切分
                    fea1 = fea.permute(0, 2,3,1)
                    fea2 = fea1[b,gj,gi]
                    fea3 = fea2
                    # 在这里得到 源域原型实例
                    for j in range(self.nc):
                        Pks = fea2[tcls[i] == j]
                        Pks = torch.sigmoid(Pks)
                        if Pks.shape[0]:
                            Pks = torch.mean(Pks, dim= 0)
                            Gpk = self.feaProtypeList[i][j]
                            a = (torch.cosine_similarity(Gpk, Pks, dim= 0) + 1) * 0.5
                            self.feaProtypeList[i][j].data = a * Pks + (1 - a) * Gpk





   # 从 target得到正样本
    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)  37*6
        device = p[0].device
        na, nt = 3, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=device)  # normalized to gridspace gain
        ai = torch.arange(na, device=device).float().view(na, 1).repeat(1, nt)  # same as .3repeat_interleave(nt) 3*n
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices 3*n*7 3*n*6 + 3*n*1 = 3*N*7
        #input targets(image,class,x,y,w,h，anchornumber)
        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=device).float() * g  # offsets

        for i in range(self.nl):   # number of layer anchor scale 也是不同的
            anchors, shape = self.anchors[i], p[i].shape               # anchor以及映射到特征图
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain 1 1 80 80 80 80 1

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)    将标签中归一化后的xywh映射到特征图上
            # targets: [img_id, cls_id, x_norm, y_norm, w_norm, h_norm, anchor_id]
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio   None 增加一个维度 在中间增加一个维度 3*2 -> 3*1*2 注意none的位置
                j = torch.max(r, 1 / r).max(2)[0] < 4.0  # compare  函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter  获取anchor与gt的宽高比值，如果比值超出anchor_t，那么该anchor就会被舍弃，不参与loss计算   n*7

                # Offsets    j,k分别表示x,y轴的条件满足
                gxy = t[:, 2:4]  # grid xy  中心点：gxy
                gxi = gain[[2, 3]] - gxy  # inverse   反转中心点：gxi   图片的长宽-中心点的值
                j, k = ((gxy % 1 < g) & (gxy > 1)).T    # 距离当前格子左上角较近的中心点，并且不是位于边缘格子内  yolo的grid坐标原点在左上角
                l, m = ((gxi % 1 < g) & (gxi > 1)).T   # 距离当前格子右下角较近的中心点，并且不是位于边缘格子内 同时进行向量转置.T
                j = torch.stack((torch.ones_like(j), j, k, l, m))       # 新的维度进行堆叠 5*n [原点，左边，上边，右边，下边]
                fzb1 = t.repeat((5, 1, 1))       # 用来测试的
                t = t.repeat((5, 1, 1))[j]                        # 5*n*7
                #offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]   # 1*n*2  5*1*2
                fzb1 = torch.zeros_like(gxy)[None]  # 1*n*2
                fzb2 = off[:, None]              #    5*1*2
                fzb3 = fzb1 + fzb2               # 5*n*2
                offsets = fzb3[j]                # ?*2   5个相对应的偏置
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()  # gij就是正样本格子的整数部分即索引
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid clamp()函数的功能将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量。
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box   取小数 这里(gxy-gij)的取值范围-0.5 ~ 1.5

            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            '''
            indices = [imageSeq,anchorSeq, 格子y索引，格子x索引
            tbox = 格子内相对位移，anchor长宽
            '''
        return tcls, tbox, indices, anch



#实例对齐网络
class TinstanceLayer(nn.Module):

    def __init__(self, detectLayer):
        super(TinstanceLayer, self).__init__()
        m = detectLayer
        self.nc = m.nc  # number of classes
        self.no = m.nc + 5 + 2  # number of outputs per anchor
        self.anchors = m.anchors
        self.nl = m.nl  # number of detection layers
        self.na = m.na # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.ROI = RoIAlign(output_size =7,spatial_scale = 1.,sampling_ratio = -1)
        self.modelLayer = nn.ModuleList()
        self.convLayer = nn.ModuleList()
        self.instance_loss = nn.BCEWithLogitsLoss()
        self.maxdetnum = 25

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid(y, x, indexing='ij')
        else:
            yv, xv = torch.meshgrid(y, x)
        pos =  torch.stack((xv, yv), 2).expand(shape)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid, pos

    def forward(self, features,x):
        z = []  # inference output
        preds = []
        ROILayer = []
        for i in range(self.nl):              # 循环每一个layer 其实可以去掉
            bs, _, ny, nx,_ = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            grid, anchor_grid, pos = self._make_grid(nx, ny, i)
            grid = grid.to(x[0].device)
            anchor_grid = anchor_grid.to(x[0].device)
            y = x[i].sigmoid()
            xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
            xy = (xy * 2 + grid) # xy
            pos = pos.expand(xy.shape)
            wh = (wh * 2) ** 2 * anchor_grid  # wh
            pos = pos.to(x[0].device)
            y = torch.cat((xy, wh, conf, pos), 4)
            z.append(y.view(bs, -1, self.no))

            # 非极大值印制  不同的layer list of detections, on (n,6) tensor per image [xyxy, conf, cls]
            pred = self.non_max_suppression_tensor(z[i], max_det=self.maxdetnum, iou_thres = 0.7)    # list 对应图片batch [n,5]  (xyxy, conf, cls) list of detections, on (n,6) tensor per image [xyxy, conf, cls]
            preds.append(pred)
            shapefea = features[i].shape
            dettensor = torch.zeros([shapefea[0],self.maxdetnum,shapefea[1]], device=x[0].device, dtype=x[0].dtype)


            for j, det in enumerate(pred):  # per image
                # det[:, :4] = scale_coords(imagesShape.shape[2:], det[:, :4], features[i].shape).round()  # 重新调整坐标 其实不需要调整
                #roiVector = self.ROI(features[i][j:j+1], [det[...,:4]])
                feature = features[i].permute(0,2,3,1)
                posX = det[:,-1].squeeze().long()
                posY = det[:,-2].squeeze().long()
                roiVector= feature[j,posX,posY]
                dettensor[j,:roiVector.shape[0]] = roiVector

            # 截取特征图  得到roi向量
            # feature[i] [batchsize,3,w,h]   detlist [batchsize,n,6]  # 提取Roi长度
            ROILayer.append(dettensor)
        return ROILayer

    def non_max_suppression_tensor(self,prediction,
                            conf_thres=0.25,
                            iou_thres=0.45,
                            classes=None,
                            agnostic=False,
                            multi_label=False,
                            labels=(),
                            max_det=300):
        """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

        Returns:
             list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        # """

        bs = prediction.shape[0]  # batch size
        output = [torch.zeros((0, 8), device=prediction.device)] * bs

        nc = prediction.shape[2] - 7  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates
        # Checks
        #xc = prediction[..., 4]
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 0.3 + 0.03 * bs  # seconds to quit after
        redundant = True  # require redundant detections
        #multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)    这里竟然会影响？  这是奇葩的bug
        merge = False  # use merge-NMS

        for xi in range(prediction.shape[0]):  # image index, image inference
            x = prediction[xi]
            # Compute conf
            x[:, 5:-2] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])
            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:-2].max(1, keepdim=True)
                pos = x[:,-2:]
                x = torch.cat((box, conf, j.float(), pos), 1)   # 从前后景的conf置信度 到 类别的置信度
                #x = torch.cat((box, conf, j.float()), 1)
            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes.clone().detach(), scores.clone().detach(), iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            output[xi] = x[i]
            # if (time.time() - t) > time_limit:
            #     LOGGER.warning(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
            #     break  # time limit exceeded

        return output
class InterToOut(nn.Module):
    def __init__(self,ch):
        super(InterToOut, self).__init__()
        self.ch = ch

    def forward(self, input):
        return input


