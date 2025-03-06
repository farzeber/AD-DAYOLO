# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
import math

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel
from torch.nn import functional as F

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                # 这里从pred挑选 选择 与 切分
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio     # 这里设置物体存在值

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp     # one-shot标签
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)   # 这里用到了正负样本
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        if math.isnan(lobj.item()):
            print("hello")                 # 当输入的pred的batch为0，会存在bug
        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)  37*6
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .3repeat_interleave(nt) 3*n
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
            device=self.device).float() * g  # offsets

        for i in range(self.nl):   # number of layer anchor scale 也是不同的
            anchors, shape = self.anchors[i], p[i].shape               # anchor以及映射到特征图
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain 1 1 80 80 80 80 1

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)    将标签中归一化后的xywh映射到特征图上
            # targets: [img_id, cls_id, x_norm, y_norm, w_norm, h_norm, anchor_id]
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio   None 增加一个维度 在中间增加一个维度 3*2 -> 3*1*2 注意none的位置
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare  函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
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

# add by fzb
def compute_domain_loss(hiddenout,domdomainLabelsainflag,device):
    # inittensor = torch.zeros(3,3)
    # loss = nn.L1Loss(inittensor,inittensor)
    loss = torch.zeros(1, device=device)
    tensorDommain = [ torch.ones_like(x,device=device)  for x in hiddenout]            # layer,batchsiz,1,w,h
    domdomainLabelsainflag = torch.tensor(domdomainLabelsainflag,device=device)
    domdomainLabelsainflag = domdomainLabelsainflag[...,None,None,None]
    tensorDommain = [x*domdomainLabelsainflag for x in tensorDommain]
    for sdmain,tdmain in zip(hiddenout,tensorDommain):
        loss += nn.BCEWithLogitsLoss()(sdmain,tdmain)      # 通道表示预测为某一类的概率
    return loss / (len(hiddenout))

def compute_instance_loss(pred, instanceLabels,device):
    instanceLabels = instanceLabels[None,...,None,None].to(device)
    instls = torch.zeros(1, device=device)  # inst loss
    model = nn.BCEWithLogitsLoss()
    tInst = torch.ones_like(pred,device=device)   # batch, seqlayer, inst, 1
    tInst *= instanceLabels
    instls += model(pred, tInst)
    return instls

def compute_ImageDomain_loss(preds,domdomainLabelsTensor,device,fakeLabel = None):
    loss = torch.zeros(1, device=device)
    for item,pred in enumerate(preds):
        if not fakeLabel is None:
            loss += compute_integdomain_loss(pred, domdomainLabelsTensor, device,fakeLabel[item])
        else:
            loss += compute_integdomain_loss(pred, domdomainLabelsTensor, device)
    return loss/ (len(preds))

def compute_clip_loss(preds,domdomainLabelsTensor,device,fakeLabel = None):
    loss = torch.zeros(1, device=device)
    for item,pred in enumerate(preds):
        if not fakeLabel is None:
            loss += compute_integdomain_loss(pred, domdomainLabelsTensor, device,fakeLabel[item])
        else:
            loss += compute_integdomain_loss(pred, domdomainLabelsTensor, device)
    return loss/ (len(preds))

def compute_ImageDomain_with_hrloss(preds,domdomainLabelsTensor,device,hrImages,domdomainLabelsTensorHr):
    loss = torch.zeros(1, device=device)
    lossCon = []
    model = nn.CrossEntropyLoss(reduce=False)
    for pred in preds:
        label = domdomainLabelsTensor.to(device)
        # 本地这段代码 真是奇怪？
        tDomain = torch.ones_like(pred, device=device, requires_grad=False)
        domdomainLabelsTensor1 = domdomainLabelsTensor[..., None, None, None].to(device)  # [batchsize,1,1]
        tDomain[:, 1:, :, :] *= domdomainLabelsTensor1
        tDomain[:, :1, :, :] = 1 - tDomain[:, 1:, :, :]
        # tDomain = tDomain.type(torch.long)
        test = model(pred, tDomain)
        lossCon.append(test)

    lossCon = lossCon[::-1]
    hrImages = hrImages[:-1]
    hrImages = hrImages[::-1]
    for pred,hr in zip(lossCon,hrImages):
        hrfilter = hr[domdomainLabelsTensorHr != 3]
        hrfilter = torch.mean(hrfilter, dim=1) + 1
        loss += torch.mean(hrfilter * pred)

    return loss

def compute_prototypeInst_loss(instSlayer, instT, device):
    loss = torch.zeros(1, device=device)
    feaProtypeList = instSlayer.feaProtypeList
    for i,value in enumerate(instT):
        value = torch.sigmoid(value)
        feaProtype = feaProtypeList[i]
        pijlist = []
        for classSvalue in feaProtype:
            shape = value.shape
            intervalue1 = torch.mv(value.view(-1,shape[2]), classSvalue)
            #intervalue2 = torch.exp(intervalue1)
            pijlist.append(intervalue1)
        sumpij = sum(pijlist)

        pijlist1 = []
        for j in pijlist:
            pij = j / (sumpij + 1e-6)
            pij = pij * torch.log(pij + 1e-6)
            pijlist1.append(-pij)
        sumpij2 = sum(pijlist1)
        test = torch.mean(sumpij2)
        if torch.isnan(test):
            fzb = 1 + 1
        loss = loss + torch.mean(sumpij2)

    return loss / (i + 1)

def compute_hr_loss(preds,hrImages,device):
    loss = torch.zeros(1, device=device)
    model = nn.SmoothL1Loss()
    preds = preds[::-1]
    hrImages = hrImages[:-1]
    hrImages = hrImages[::-1]
    for pred,hr in zip(preds,hrImages):
        loss += model(pred,hr)
    return loss

def compute_hr_loss_STdomain(preds,hrImages,device,domainLabelsTensor):
    loss = torch.zeros(1, device=device)
    model = nn.SmoothL1Loss()
    preds = preds[::-1]
    hrImages = hrImages[:-1]
    hrImages = hrImages[::-1]
    for pred,hr in zip(preds,hrImages):
        predfilter = pred[domainLabelsTensor != 3]
        hrfilter = hr[domainLabelsTensor != 3]
        loss += model(predfilter,hrfilter)
    return loss

def compute_integdomain_loss(pred,domdomainLabelsTensor,device, fakeLabel = None):

    loss = torch.zeros(1, device=device)
    model = nn.CrossEntropyLoss()

    label = domdomainLabelsTensor.to(device)
    # 服务器上用这段代码
    # pred [batchsize,2,d,d]
    # shapes = pred.shape
    # tDomain = torch.ones(shapes[0],shapes[2],shapes[3],device=device, dtype=torch.long)       # pred [batchsize,d,d]
    # domdomainLabelsTensor = domdomainLabelsTensor[...,None,None].to(device) # [batchsize,1,1]
    # tDomain *= domdomainLabelsTensor
    # test = model(pred, tDomain)
    # loss += model(pred, tDomain)

    # 本地这段代码 真是奇怪？通道0源域，通道1目标域
    tDomain = torch.ones_like(pred,device=device, requires_grad= False)
    domdomainLabels = domdomainLabelsTensor
    domdomainLabelsTensor = domdomainLabelsTensor[...,None,None,None].to(device) # [batchsize,1,1]
    tDomain[:,1:,:,:] = tDomain[:,1:,:,:] * domdomainLabelsTensor
    tDomain[:,:1,:,:] = 1 - tDomain[:,1:,:,:]
    if torch.is_tensor(fakeLabel):
        tDomain[domdomainLabels == 3] = fakeLabel[domdomainLabels == 3]
    else:
        pred = pred[domdomainLabels != 3]
        tDomain = tDomain[domdomainLabels != 3]

    #tDomain = tDomain.type(torch.long)
    test = model(pred, tDomain)
    loss += test

    # temp1 = torch.exp(pred) / torch.sum(torch.exp(pred), dim=1, keepdim=True)
    #
    # # log
    # temp2 = torch.log(temp1)
    #
    # # nll
    # #temp3 = torch.gather(temp2, dim=1, index=label.view(-1, 1))
    # temp4 =  nn.NLLLoss()(temp2, label)
    # output = torch.mean(temp4)

    return loss

def compute_priorAndImageDa_loss(priorsZ, penOut, domdomainLabelsTensor,device):

    # 1. priorsZ是原图片大小的，需要插值    用同态滤波得到
    # 2. penOut 包括三张不同大小的图片输出
    # 3. L2 图片loss  nn.MSELoss
    l2model = nn.MSELoss()
    p8 = F.interpolate(priorsZ ,scale_factor=0.125,mode="bilinear")
    p16 = F.interpolate(p8 ,scale_factor=0.5,mode="bilinear")
    loss1 = l2model(p8, penOut[0])
    loss2 = l2model(p16, penOut[1])
    loss3 = compute_integdomain_loss(penOut[2], domdomainLabelsTensor, device)
    loss = loss1 + loss2 + loss3
    return loss

def compute_consis_loss(insts, images, device):
    model = nn.MSELoss(size_average=False)
    # loss = torch.zeros(1, device=device)
    sum = torch.zeros(1, device=device,requires_grad=False)
    images = images.detach()
    with torch.no_grad:
        for x in images:
            y = nn.Sigmoid(x)
            sum += torch.mean(y)
        sum = sum / (len(images))

    insts = insts.view(-1,1)       # batch, seqlayer, inst, 1 -> inst, 1
    sum = sum.repeat(insts.size())
    return model(insts, sum)


def compute_mLabel_loss(outputS,outputT,domainLabelsTensor,targets,device):
    filteroutPutS = outputS[domainLabelsTensor == 0]  #筛选出源域
    nllModel = nn.BCEWithLogitsLoss()
    soothLoss = nn.SmoothL1Loss(reduction='none')
    # 生成多分类标签
    label = torch.zeros_like(filteroutPutS, device=device)
    objs = targets[:,:2]   # 显示类别值 和 样本值
    for i in objs:
        label[int(i[0])][int(i[1])] = 1

    # 源域多标签loss1
    loss1 = nllModel(filteroutPutS, label)

    loss2 = soothLoss(torch.sigmoid(outputT), torch.sigmoid(outputS.detach()))
    loss2S = loss2[domainLabelsTensor == 0]
    loss2T = loss2[domainLabelsTensor == 1]
    loss2 = torch.mean(loss2S) - torch.mean(loss2T)
    return loss1 + loss2

def compute_smLabel_loss(outputS,domainLabelsTensor,targets,device):
    filteroutPutS = outputS[domainLabelsTensor == 0]  #筛选出源域
    nllModel = nn.BCEWithLogitsLoss()
    soothLoss = nn.SmoothL1Loss(reduction='none')
    # 生成多分类标签
    label = torch.zeros_like(filteroutPutS, device=device)
    objs = targets[:,:2]   # 显示类别值 和 样本值
    for i in objs:
        label[int(i[0])][int(i[1])] = 1

    # 源域多标签loss1
    loss1 = nllModel(filteroutPutS, label)
    return loss1

def compute_pen_loss(priorsZ, penOut):

    # 1. priorsZ是原图片大小的，需要插值    用同态滤波得到
    # 2. penOut 包括三张不同大小的图片输出
    # 3. L2 图片loss  nn.MSELoss
    l2model = nn.SmoothL1Loss()
    p8 = F.interpolate(priorsZ ,scale_factor=0.125,mode="bilinear")
    p16 = F.interpolate(p8 ,scale_factor=0.5,mode="bilinear")
    p32 =F.interpolate(p16 ,scale_factor=0.5,mode="bilinear")
    loss1 = l2model(p8, penOut[0])
    loss2 = l2model(p16, penOut[1])
    loss3 = l2model(p32, penOut[2])
    loss = loss1 + loss2 + loss3
    return loss

def compute_integpen_loss(priorsZ, penOut):

    # 1. priorsZ是原图片大小的，需要插值    用同态滤波得到
    # 2. penOut 包括三张不同大小的图片输出
    # 3. L2 图片loss  nn.MSELoss
    l2model = nn.MSELoss()
    p8 = F.interpolate(priorsZ ,scale_factor=0.125,mode="bilinear")
    p16 = F.interpolate(p8 ,scale_factor=0.5,mode="bilinear")
    p32 =F.interpolate(p16 ,scale_factor=0.5,mode="bilinear")
    loss3 = l2model(p32, penOut)
    return loss3

def compute_mmd_loss(mmdFbefore, mmdFend, domainLabelsTensor):

    # 1. priorsZ是原图片大小的，需要插值    用同态滤波得到
    # 2. penOut 包括三张不同大小的图片输出
    # 3. L2 图片loss  nn.MSELoss
    l2model = MMDLoss()
    target = mmdFend[domainLabelsTensor == 1]
    source = mmdFend[domainLabelsTensor == 0]
    loss = l2model(source, target)
    return loss

class MMDLoss(nn.Module):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    '''
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2)),int(total.size(3)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2)), int(total.size(3)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss

def compute_Dinstacnce_loss(domainLabel,instacePred,mLCoutput,preds,n_class):
    # 如何计算距离d
    # x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
    # pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)   这里选的最后一层
    # tgt_instance_loss = nn.BCELoss(
    #     weight=torch.Tensor(target_weight).view(-1, 1).cuda()
    # )
    loss = torch.zeros(1, device=instacePred[0].device)
    mLCoutput = mLCoutput.detach()
    # 这里的 d用标量计算

    for i, pred in enumerate(preds):
        pred = pred.detach()
        with torch.no_grad():
            _,_,conf,plcs = pred.split((2, 2, 1, n_class),4)
            plcs = conf * plcs
            shape = plcs.shape
            plcs = plcs.reshape(shape[0], -1, shape[4])
            plcs,_ = torch.max(plcs,dim=1)             # 每个类的 最大框值
            d = torch.abs(torch.sigmoid(mLCoutput) - torch.sigmoid(plcs))      # 这里需要使用sigmoid函数隐射到 0 1 上
            d = torch.mean(d,dim=1)
            d[domainLabel == 0] = 1.0
        model = nn.BCEWithLogitsLoss(
            weight= d.view(-1,1)
        )
        loss = loss + model(instacePred[i], 1.0 * domainLabel.view(-1,1).to(loss.device))

    return loss / len(preds)

def compute_instVector_loss(instVectorList, device):
    loss = torch.zeros(1, device=device)
    model = nn.L1Loss()
    for instValue in instVectorList:
        x = torch.mean(instValue)

        loss = loss + model(x, torch.ones_like(x, device=device)*0.9)

    return loss / len(instVectorList)