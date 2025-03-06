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
class ganInstLayer(nn.Module):
    def __init__(self,nc = 80,anchors = (),inCh =()):
        super().__init__()
        self.nc = nc
        self.nl = len(anchors)  # number of detection layers
        self.feaProtypeList = torch.nn.ParameterList()
        self.ctoModelList = torch.nn.ModuleList()
        for ch in inCh:
            # 用来存储原型变量的
            fea = torch.nn.ParameterList([torch.nn.Parameter(data=torch.zeros([ch],dtype=torch.float), requires_grad=False) for j in range(nc)])
            self.feaProtypeList.append(fea)
            self.ctoModelList.append(nn.Sequential(nn.Conv2d(ch,ch,1,1,0),nn.ReLU(),nn.Conv2d(ch,ch,1,1,0)))

        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)

    def forward(self, feature, target, domainlables):
        #self.forward_ori(feature, target, domainlables) 暗光正常光相似度 + 有监督对比损失

        #self.forward_ori(feature, target, domainlables) 暗光正常光相似度 + 无监督对比损失 + 只有一对正样本 只在同类box中进行比较
        # 原型对齐加实例
        return self.forward_part_contro(feature, target, domainlables)



    def forward_ori(self,feature,target,domainlables):
        # target为空？
        if not torch.is_tensor(target):
            return

        if not isinstance(domainlables, list):
            return
        tcls, _, indices, _ = self.build_targets(feature,target)
        #  交换源域中 两个标注的域feature顺序
        feaN = feature[0].shape[0]
        feaNlist = list(range(feaN))
        for i,lable in enumerate(domainlables):
            if lable == 3:
                temp = feaNlist[i - 1]
                feaNlist[i -1] = feaNlist[i]
                feaNlist[i] = temp

        #得到余弦相似度
        cosList = []
        darkList = []
        for i,fea in enumerate(feature):
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                # 这里从pred挑选 选择 与 切分
                fea = self.ctoModelList[i](fea)
                # 源域的向量
                fea1 = fea.permute(0, 2,3,1)
                fea2 = fea1[b,gj,gi]

                # 在这里得到 向量余弦相似度， 暗光与正常光实例
                fea1ba = fea1[feaNlist]
                fea2ba = fea1ba[b,gj,gi]

                costy = torch.cosine_similarity(fea2,fea2ba, dim=1)
                #torch.minimum(costy, torch.tensor([0.95],device=costy.device))
                # 暗光实例和 正常光实例对齐
                cosList.append(costy)

                #有监督对比学习，相同类的向量聚集在一起，但是一个batch中可能不是所有类的实例都存在，所以我们需要引入 原型实例
                ################################## 在这里得到 源域原型实例
                # Gpkvec = []
                # Gpkclass = []
                # with torch.no_grad():
                #     for j in range(self.nc):
                #         Pks = fea2[tcls[i] == j]
                #         #Pks = torch.sigmoid(Pks)
                #         if Pks.shape[0]:
                #             Pks = torch.mean(Pks, dim=0)
                #             Gpk = self.feaProtypeList[i][j]
                #             a = (torch.cosine_similarity(Gpk, Pks, dim=0) + 1) * 0.5
                #             self.feaProtypeList[i][j].data = a * Pks + (1 - a) * Gpk
                #
                #         Gpkvec.append(self.feaProtypeList[i][j].data)
                #         Gpkclass.append(j)
                # Gpkvec = torch.stack(Gpkvec,dim=0)
                # Gpkclass = torch.tensor(Gpkclass, device=fea2.device)
                # ################################## 加入原型实例end
                # fea2 = torch.cat([fea2, Gpkvec], dim= 0)
                # tcls[i] = torch.cat([tcls[i], Gpkclass], dim=0)
                darkList.append((fea2, tcls[i]))
        return cosList,darkList

    def forward_wujiandu(self,feature,target,domainlables):
        # target为空？
        if not torch.is_tensor(target):
            return

        if not isinstance(domainlables, list):
            return
        targetseq = target[:,0]
        target = target[targetseq % 2 == 0]

        tcls, _, indices, _ = self.build_targets(feature,target)
        #  交换源域中 两个标注的域feature顺序
        feaN = feature[0].shape[0]
        feaNlist = list(range(feaN))
        for i,lable in enumerate(domainlables):
            if lable == 3:
                temp = feaNlist[i - 1]
                feaNlist[i -1] = feaNlist[i]
                feaNlist[i] = temp

        #得到余弦相似度
        cosList = []
        darkList = []
        for i,fea in enumerate(feature):
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                # 这里从pred挑选 选择 与 切分
                fea = self.ctoModelList[i](fea)

                fea1 = fea.permute(0, 2,3,1)
                fea2 = fea1[b,gj,gi]

                # 在这里得到 向量余弦相似度， 暗光与正常光实例
                fea1ba = fea1[feaNlist]
                fea2ba = fea1ba[b,gj,gi]
                costy = torch.cosine_similarity(fea2,fea2ba, dim=1)
                torch.minimum(costy, torch.tensor([0.9],device=costy.device))
                # 暗光实例和 正常光实例对齐
                cosList.append(costy)
                for j in range(self.nc):
                    Pks = fea2[tcls[i] == j]
                    Pks1 = fea2ba[tcls[i] == j]

                    if Pks.shape[0] !=0:
                        Pks2 = torch.cat([Pks, Pks1], dim= 0)
                        Pklabels = [range(Pks.shape[0])] + [range(Pks.shape[0])]
                        darkList.append((Pks2, torch.tensor(Pklabels)))
                #darkList.append((fea2, tcls[i]))
        return cosList,darkList

    def forward_proto_contro(self,feature,target,domainlables):
        # target为空？
        if not torch.is_tensor(target):
            return

        if not isinstance(domainlables, list):
            return
        targetseq = target[:,0]
        # 这里过滤出和源域一样的目标域target
        target = target[targetseq % 2 == 0]

        tcls, _, indices, _ = self.build_targets(feature,target)
        #  交换源域中 两个标注的域feature顺序
        feaN = feature[0].shape[0]
        feaNlist = list(range(feaN))
        for i,lable in enumerate(domainlables):
            if lable == 3:
                temp = feaNlist[i - 1]
                feaNlist[i -1] = feaNlist[i]
                feaNlist[i] = temp

        #得到余弦相似度
        cosList = []
        darkList = []

        for i,fea in enumerate(feature):
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            n = b.shape[0]  # number of targets
            # 用了感知机转换了空间
            fea = self.ctoModelList[i](fea)
            fea1 = fea.permute(0, 2, 3, 1)
            fea1ba = fea1[feaNlist]
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                # 这里从pred挑选 选择 与 切分
                fea2 = fea1[b,gj,gi]
                # 在这里得到 向量余弦相似度， 暗光与正常光实例
                fea2ba = fea1ba[b,gj,gi]
                costy = torch.cosine_similarity(fea2,fea2ba, dim=1)
                #torch.minimum(costy, torch.tensor([1.0],device=costy.device))
                # 暗光实例和 正常光实例对齐
                cosList.append(costy)
                # 有监督原型对齐， 一对正样本，同时缩小负样本，同时更新全局原型实例
                Gpkvec = []
                Gpkclass = []
                aInfo = []
                for j in range(self.nc):
                    Pks = fea2[tcls[i] == j]
                    Pks1 = fea2ba[tcls[i] == j]
                    if Pks.shape[0] !=0:
                        Pks2 = torch.cat([Pks, Pks1], dim= 0)
                        Pksproto = torch.mean(Pks,dim=0)
                        Pks1proto = torch.mean(Pks1,dim=0)
                        #用来更新全局实例
                        with torch.no_grad():
                            PksG = torch.mean(Pks2, dim=0)
                            Gpk = self.feaProtypeList[i][j]
                            # ema 平均移动
                            a = (torch.cosine_similarity(Gpk, PksG, dim=0) + 1) * 0.5
                            self.feaProtypeList[i][j].data = (1 -a) * PksG + (a) * Gpk
                            aInfo.append(a)
                        # Pklabels = [j] + [j]
                        # darkList.append((Pks2, torch.tensor(Pklabels)))
                        Gpkvec = Gpkvec + [Pksproto, Pks1proto]
                        Gpkclass = Gpkclass + [j]*2
                    else:
                        Gpkvec.append(self.feaProtypeList[i][j].data)
                        Gpkclass.append(j)
                        aInfo.append(0.)
                LOGGER.info(' '.join(map(str, aInfo)))
                darkList.append((torch.stack(Gpkvec, dim=0), torch.tensor(Gpkclass, device=fea.device)))
        return cosList,darkList

    def forward_zhengfu_contro(self,feature,target,domainlables):
        # target为空？
        if not torch.is_tensor(target):
            return

        if not isinstance(domainlables, list):
            return

        domainTensor = torch.tensor(domainlables)

        #targetseq = target[:,0]
        # 这里过滤出和源域一样的目标域target,  因为源域和目标域有重复标注
        #target = target[targetseq % 2 == 0]

        tcls, _, indices, _ = self.build_targets(feature,target)
        #  交换源域中 两个标注的域feature顺序
        feaN = feature[0].shape[0]
        feaNlist = list(range(feaN))
        for i,lable in enumerate(domainlables):
            if lable == 3:
                temp = feaNlist[i - 1]
                feaNlist[i -1] = feaNlist[i]
                feaNlist[i] = temp

        #得到余弦相似度
        cosList = []
        darkList = []

        for i,fea in enumerate(feature):
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            n = b.shape[0]  # number of targets
            #fea = self.ctoModelList[i](fea)

            fea1 = fea.permute(0, 2, 3, 1)
            fea1ba = fea1[feaNlist]
            # 去除目标域，只保留源域，协助域
            fea1 = fea1[domainTensor != 1]
            fea1ba = fea1ba[domainTensor != 1]

            feacos = torch.cosine_similarity(fea1,fea1ba, dim = 3)
            #feacos = 0 - feacos
            # if n:
            #     # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
            #     # 这里从pred挑选 选择 与 切分
            #     feacos[b,gi,gi] = 0 - feacos[b,gi,gi]
                # 暗光实例和 正常光实例对齐
            cosresult = feacos.view(-1)
            cosList.append(cosresult)
        return cosList,darkList

    def forward_part_contro(self,feature,target,domainlables):
        # target为空？
        if not torch.is_tensor(target):
            return
        if not isinstance(domainlables, list):
            return
        targetseq = target[:,0]
        # 这里过滤出和源域一样的目标域target
        target = target[targetseq % 2 == 0]
        tcls, _, indices, _ = self.build_targets(feature,target)

        #  交换源域中 两个标注的域feature顺序
        feaN = feature[0].shape[0]
        feaNlist = list(range(feaN))
        for i,lable in enumerate(domainlables):
            if lable == 3:
                temp = feaNlist[i - 1]
                feaNlist[i -1] = feaNlist[i]
                feaNlist[i] = temp

        #得到余弦相似度
        cosList = []
        darkList = []

        for i,fea in enumerate(feature):
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tcls[i] = tcls[i].detach()
            n = b.shape[0]  # number of targets

            # 用了感知机转换了空间
            fea = self.ctoModelList[i](fea)
            fea1 = fea.permute(0, 2, 3, 1)
            fea1ba = fea1[feaNlist]
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                # 这里从pred挑选 选择 与 切分
                fea2 = fea1[b,gj,gi]
                # 在这里得到 向量余弦相似度， 暗光与正常光实例
                fea2ba = fea1ba[b,gj,gi]
                costy = torch.cosine_similarity(fea2,fea2ba, dim=1)
                #torch.minimum(costy, torch.tensor([1.0],device=costy.device))
                # 暗光实例和 正常光实例对齐
                cosList.append(costy)
                # 有监督原型对齐， 一对正样本，同时缩小负样本，同时更新全局原型实例
                Gpkvec = []
                Gpkclass = []
                aInfo = []

                for j in range(self.nc):
                    Pks = fea2[tcls[i] == j]
                    Pks1 = fea2ba[tcls[i] == j]
                    if Pks.shape[0] !=0:
                        Pks2 = torch.cat([Pks, Pks1], dim= 0)
                        #取最近的局部原型
                        with torch.no_grad():
                            PksG = torch.mean(Pks2, dim=0)
                            self.feaProtypeList[i][j].data = PksG
                        # Pklabels = [j] + [j]
                        # darkList.append((Pks2, torch.tensor(Pklabels)))
                        # Gpkvec = Gpkvec + [Pksproto, Pks1proto]
                        # Gpkclass = Gpkclass + [j]*2
                    else:
                        Gpkvec.append(self.feaProtypeList[i][j].data)
                        Gpkclass.append(j)
                #LOGGER.info(' '.join(map(str, aInfo)))
                positive_sample = torch.cat([fea2,fea2ba],dim=0)
                #positive_sample_label = tcls[i].repeat(2)
                positive_sample_label = torch.cat([tcls[i],tcls[i]], dim=0)
                # print(positive_sample)
                # print(positive_sample_label)
                if len(Gpkvec) != 0:
                    Gpkvec_tensor = torch.stack(Gpkvec, dim=0)
                    Gpkclass_tensor = torch.tensor(Gpkclass, device=fea.device)
                    positive_sample = torch.cat([positive_sample, Gpkvec_tensor], dim=0)
                    positive_sample_label = torch.cat([positive_sample_label,Gpkclass_tensor],dim=0)
                darkList.append((positive_sample, positive_sample_label))
        return cosList,darkList




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


# 三个域，白天加黑夜应该得到 中间域
class ganInstRtinexLayer(nn.Module):
    def __init__(self,anchors = ()):
        super().__init__()
        self.nl = len(anchors)  # number of detection layers
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)

    def forward(self,feature,target,domainlables):
        # target为空？
        if not torch.is_tensor(target):
            return

        if not isinstance(domainlables, list):
            return
        cosList = []
        dlbtensor = torch.tensor(domainlables)
        for i, fea in enumerate(feature):
            fea1 = fea.permute(0,2,3,1)
            feaR = fea[dlbtensor == 0]
            feaL = fea[dlbtensor == 3]
            feaLt = torch.tanh(feaL)
            feaI = fea[dlbtensor == 1]
            costy = torch.cosine_similarity(feaR , feaL, dim=3)
            cosList.append(costy)
        return cosList

