'''
@Project ：yolov5-master 
@File    ：oneToOneInstLayer.py
@IDE     ：PyCharm 
@Author  ：付卓彬
@Date    ：2024/4/17 20:44 
'''
import torch.nn as nn
import torch
from torchvision.ops import RoIAlign
from utils.general import xywh2xyxy
from models.common import  Conv
from torch.autograd import Function
from typing import Any, Optional, Tuple

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

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

class oneToOneInstLayer(nn.Module):
    def __init__(self,inputc):
        super(oneToOneInstLayer, self).__init__()
        # 不同层次的特征图
        #hard_code = 64
        # spatial_scale: float = 1.0,
        # sampling_ratio: int = -1,
        self.ROIL = RoIAlign(output_size =1,spatial_scale =1.0, sampling_ratio=-1,aligned=True)
        self.ROIM = RoIAlign(output_size =1,spatial_scale =1.0, sampling_ratio=-1,aligned=True)
        self.ROIH = RoIAlign(output_size =1,spatial_scale =1.0, sampling_ratio=-1,aligned=True)
        self.ROIList = torch.nn.ModuleList([self.ROIL, self.ROIM, self.ROIH])

        # 下面可能优点笨重，可以先用卷积进行通道江维度，再用全连接层进行连接
        # self.classifierL = nn.Sequential(
        #     nn.Linear(128*5*5, hard_code),
        #     nn.LeakyReLU(),
        #     nn.Linear(hard_code, hard_code),
        # )
        # self.classifierM = nn.Sequential(
        #     nn.Linear(256*5*5, hard_code),
        #     nn.LeakyReLU(),
        #     nn.Linear(hard_code ,hard_code),
        # )
        # self.classifierH = nn.Sequential(
        #     nn.Linear(512*5*5, hard_code),
        #     nn.LeakyReLU(),
        #     nn.Linear(hard_code, hard_code),
        # )
        # self.FCList = torch.nn.ModuleList([self.classifierL, self.classifierM, self.classifierH])
        self.sloss1 = nn.CrossEntropyLoss()
        self.dloss = nn.SmoothL1Loss()
        self.l2norm = Normalize(2) # 把向量单位化
        self.nce_T = 0.07    #temperature for NCE loss
        # self.F1 = nn.Sequential(Conv(inputc[0],128,3,1), Conv(128,32,3,1), Conv(32,32,3,2), Conv(32,32,3,2))
        # self.F2 = nn.Sequential(Conv(inputc[1],256,3,1), Conv(256,64,3,1),Conv(64,32,3,1),Conv(32,32,3,2))
        # self.F3 = nn.Sequential(Conv(inputc[2],512,3,1), Conv(512,256,3,1), Conv(256,128,3,1),Conv(128,32,3,1))
        #
        # self.convQ = Conv(96,96,1,1)
        # self.convK = Conv(96,96,1,1)
        # self.convV = Conv(96,96,1,1)
        #
        # self.F123 = nn.Sequential(Conv(96, 32, 3, 1))

    def forward(self, feature, target, domainlables):

        # 对比损失 变分类
        #output = self.forward_One_to_One(feature, target, domainlables)
        output = self.forwardWithContro(feature, target, domainlables)
        return output

    def forward_ori(self, feature, target, domainlables):
        if not torch.is_tensor(target):
            return

        if not isinstance(domainlables, list):
            return

        feaN = feature[0].shape[0]
        feaNlist = list(range(feaN))
        for i,lable in enumerate(domainlables):

            if lable == 3 :
                assert (lable == 3 and domainlables[i - 1] == 0), '不能交换源域和协助域的顺序'
                temp = feaNlist[i - 1]
                feaNlist[i -1] = feaNlist[i]
                feaNlist[i] = temp
        n = target.shape[0]

        # 保存roi特征图
        loss = torch.zeros(1, device=feature[0].device)
        if n:
            for i, fea in enumerate(feature):
                xyxytarget = xywh2xyxy(target[:, 2:] * (feature[i].shape[2]))
                bxyxytarget = torch.concat([target[:, 1:2], xyxytarget], dim=1)

                feaswap = fea[feaNlist]
                # 粗略的对齐 1
                sfeaRoi = self.ROIList[i](fea, bxyxytarget)
                sshape = sfeaRoi.shape
                sfeaLabel = torch.zeros(sshape[0], device=sfeaRoi.device, dtype=torch.long)

                afeaRoi = self.ROIList[i](feaswap, bxyxytarget)
                ashape = afeaRoi.shape
                afeaLabel = torch.ones(ashape[0], device=afeaRoi.device, dtype=torch.long)

                sfeadc = self.FCList[i](sfeaRoi.view(sshape[0], -1))
                afeadc = self.FCList[i](afeaRoi.view(ashape[0], -1))

                lossS = self.sloss1(sfeadc, sfeaLabel)  # 让源域 在实例上 靠近协助域
                lossA = self.sloss1(afeadc, afeaLabel)
                loss = loss + (lossS+lossA)/2
        return loss


    def forward_ronghe(self, feature, target, domainlables):
        #self.forward_ori(feature, target, domainlables) 暗光正常光相似度 + 有监督对比损失

        #self.forward_ori(feature, target, domainlables) 暗光正常光相似度 + 无监督对比损失 + 只有一对正样本 只在同类box中进行比较
        # 原型对齐加实例
        if not torch.is_tensor(target):
            return

        if not isinstance(domainlables, list):
            return

        # 得到特征图
        feature1, feature2, feature3 = feature
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

        feaN = output.shape[0]
        feaNlist = list(range(feaN))
        for i,lable in enumerate(domainlables):

            if lable == 3 :
                assert (lable == 3 and domainlables[i - 1] == 0), '不能交换源域和协助域的顺序'
                temp = feaNlist[i - 1]
                feaNlist[i -1] = feaNlist[i]
                feaNlist[i] = temp
        n = target.shape[0]
        loss = torch.zeros(1, device=feature[0].device)
        if n:
            xyxytarget = xywh2xyxy(target[:,2:]*(output.shape[2]))
            bxyxytarget = torch.concat([target[:,1:2], xyxytarget], dim= 1)
            feaswap = output[feaNlist]
            # 粗略的对齐 1
            sfeaRoi = self.ROIM(output, bxyxytarget)
            afeaRoi = self.ROIM(feaswap, bxyxytarget)
            loss = loss + self.sloss1(sfeaRoi, afeaRoi.detach())  # 让源域 在实例上 靠近协助域

        #  交换源域中 两个标注的域feature顺序,  默认辅助域在源域后面的
        # feaN = feature[0].shape[0]
        # feaNlist = list(range(feaN))
        # for i,lable in enumerate(domainlables):
        #
        #     if lable == 3 :
        #         assert (lable == 3 and domainlables[i - 1] == 0), '不能交换源域和协助域的顺序'
        #         temp = feaNlist[i - 1]
        #         feaNlist[i -1] = feaNlist[i]
        #         feaNlist[i] = temp
        # n = target.shape[0]
        #
        # # 保存roi特征图
        # roiFeatureList = []
        # loss = torch.zeros(1, device=feature[0].device)
        # if n:
        #     xyxytarget = xywh2xyxy(target[:,2:]*(feature[0].shape[2]))
        #     bxyxytarget = torch.concat([target[:,1:2], xyxytarget], dim= 1)
        #     for i, fea in enumerate(feature):
        #         feaswap = fea[feaNlist]
        #         # 粗略的对齐 1
        #         sfeaRoi = self.ROIList[i](fea, bxyxytarget)
        #         afeaRoi = self.ROIList[i](feaswap, bxyxytarget)
        #         loss = loss + self.sloss1(sfeaRoi, afeaRoi.detach())  # 让源域 在实例上 靠近协助域

                # 更细致的对齐2 begin
                # chavalue = fea - feaswap.detach()    # 协助域不变，源域向协助域对齐
                # feaRoi = self.ROIList[i](chavalue, bxyxytarget)
                # feaRoiLabel = torch.zeros_like(feaRoi)
                # tempLoss = self.sloss1(feaRoi, feaRoiLabel)
                #
                # loss = loss + tempLoss

                # 余弦损失
                # sfeaRoiInst = sfeaRoi.view(sfeaRoi.shape[0], -1)
                # afeaRoiInst = afeaRoi.detach().view(afeaRoi.shape[0], -1)
                # costy = torch.cosine_similarity(sfeaRoiInst,afeaRoiInst, dim=1)
                # costyLabel = torch.ones_like(costy)
                # loss = loss + self.sloss1(costy, costyLabel)

        return loss

    def forward_One_to_One(self, feature, target, domainlables):
        if not torch.is_tensor(target):
            return

        if not isinstance(domainlables, list):
            return

        feaN = feature[0].shape[0]
        feaNlist = list(range(feaN))
        for i,lable in enumerate(domainlables):
            if lable == 3 :
                assert (lable == 3 and domainlables[i - 1] == 0), '不能交换源域和协助域的顺序'
                temp = feaNlist[i - 1]
                feaNlist[i -1] = feaNlist[i]
                feaNlist[i] = temp
        n = target.shape[0]

        # 保存roi特征图
        loss = torch.zeros(1, device=feature[0].device)
        n_fea = len(feature)
        if n:
            for i, fea in enumerate(feature):
                xyxytarget = xywh2xyxy(target[:, 2:] * (feature[i].shape[2]))
                bxyxytarget = torch.concat([target[:, :1], xyxytarget], dim=1)

                feaswap = fea[feaNlist]
                # 粗略的对齐 1
                sfeaRoi = self.ROIList[i](fea, bxyxytarget)
                afeaRoi = self.ROIList[i](feaswap, bxyxytarget)
                loss = loss + self.dloss(afeaRoi, sfeaRoi.detach())  # 让源域 在实例上 靠近协助域
        return loss / n_fea

    def forwardWithContro(self, feature, target, domainlables):
        if not torch.is_tensor(target):
            return

        if not isinstance(domainlables, list):
            return

        feaN = feature[0].shape[0]
        feaNlist = list(range(feaN))
        for i,lable in enumerate(domainlables):
            if lable == 3 :
                assert (lable == 3 and domainlables[i - 1] == 0), '不能交换源域和协助域的顺序'
                temp = feaNlist[i - 1]
                feaNlist[i -1] = feaNlist[i]
                feaNlist[i] = temp
        n = target.shape[0]

        # 保存roi特征图
        #loss = torch.zeros(1, device=feature[0].device)
        loss = 0.0
        n_fea = len(feature)
        sfeadc_norm_list = []
        afeadc_norm_list = []
        if n:
            for i, fea in enumerate(feature):
                xyxytarget = xywh2xyxy(target[:, 2:] * (feature[i].shape[2]))
                bxyxytarget = torch.concat([target[:, :1], xyxytarget], dim=1)
                feaswap = fea[feaNlist]

                # 粗略的对齐 1
                sfeaRoi = self.ROIList[i](fea, bxyxytarget)
                sshape = sfeaRoi.shape
                #sfeaLabel = torch.zeros(sshape[0], device=sfeaRoi.device, dtype=torch.long)

                afeaRoi = self.ROIList[i](feaswap, bxyxytarget)
                ashape = afeaRoi.shape
                #afeaLabel = torch.ones(ashape[0], device=afeaRoi.device, dtype=torch.long)

                # sfeadc = self.FCList[i](sfeaRoi.permute(0, 2, 3, 1).flatten(0, 2))
                # afeadc = self.FCList[i](afeaRoi.permute(0, 2, 3, 1).flatten(0, 2))
                # sfeadc = self.FCList[i](sfeaRoi.flatten(start_dim=1))
                # afeadc = self.FCList[i](afeaRoi.flatten(start_dim=1))
                sfeadc = sfeaRoi.view(sshape[0], -1)
                afeadc = afeaRoi.view(ashape[0], -1)
                sfeadc_norm = self.l2norm(sfeadc)
                afeadc_norm = self.l2norm(afeadc)

                # 计算不同的域靠近
                sfeadc_norm_list.append(sfeadc_norm)
                afeadc_norm_list.append(afeadc_norm)
                loss = loss + self.calculate_NCE_loss(afeadc_norm, sfeadc_norm, target[:, 1])  # 让源域 在实例上 靠近协助域
                #loss = loss + self.dloss(sfeaRoi, afeaRoi.detach())  # 让源域 在实例上 靠近协助域
            # for s_norm in sfeadc_norm_list :
            #     for a_norm in afeadc_norm_list:
            #         loss = loss + self.calculate_NCE_loss(s_norm, a_norm, target[:, 1])
            # n_fea = len(sfeadc_norm_list)*len(sfeadc_norm_list)
            del sfeadc_norm_list
            del afeadc_norm_list
        return loss / (n_fea)


    def forwardWithSubContro(self, feature, target, domainlables):
        if not torch.is_tensor(target):
            return

        if not isinstance(domainlables, list):
            return

        feaN = feature[0].shape[0]
        feaNlist = list(range(feaN))
        for i,lable in enumerate(domainlables):
            if lable == 3 :
                assert (lable == 3 and domainlables[i - 1] == 0), '不能交换源域和协助域的顺序'
                temp = feaNlist[i - 1]
                feaNlist[i -1] = feaNlist[i]
                feaNlist[i] = temp
        n = target.shape[0]

        # 保存roi特征图
        #loss = torch.zeros(1, device=feature[0].device)
        loss = 0.0
        n_fea = len(feature)
        sfeadc_norm_list = []
        afeadc_norm_list = []
        if n:
            for i, fea in enumerate(feature):
                xyxytarget = xywh2xyxy(target[:, 2:] * (feature[i].shape[2]))
                bxyxytarget = torch.concat([target[:, :1], xyxytarget], dim=1)
                feaswap = fea[feaNlist]

                # 粗略的对齐 1
                sfeaRoi = self.ROIList[i](fea, bxyxytarget)
                sshape = sfeaRoi.shape
                #sfeaLabel = torch.zeros(sshape[0], device=sfeaRoi.device, dtype=torch.long)

                afeaRoi = self.ROIList[i](feaswap, bxyxytarget)
                ashape = afeaRoi.shape
                #afeaLabel = torch.ones(ashape[0], device=afeaRoi.device, dtype=torch.long)

                sfeadc = self.FCList[i](sfeaRoi.view(sshape[0], -1))
                afeadc = self.FCList[i](afeaRoi.view(ashape[0], -1))

                # sfeadc_norm = self.l2norm(sfeadc)
                # afeadc_norm = self.l2norm(afeadc)

                # 计算不同的域靠近
                sfeadc_norm_list.append(sfeadc)
                afeadc_norm_list.append(afeadc.detach())
                #loss = loss + self.calculate_NCE_loss(sfeadc_norm, afeadc_norm, target[:, 1])  # 让源域 在实例上 靠近协助域

            # for s_norm in sfeadc_norm_list :
            #     for a_norm in afeadc_norm_list:
            #         loss = loss + self.calculate_NCE_loss(s_norm, a_norm, target[:, 1])
            # n_fea = len(sfeadc_norm_list)*len(sfeadc_norm_list)
            alllist = sfeadc_norm_list+afeadc_norm_list
            all_feature = torch.stack(sfeadc_norm_list+afeadc_norm_list, dim=1)
            all_lable = torch.stack([torch.arange(n,device=all_feature.device) for x in range(len(alllist)) ], dim=1)
            all_feature = all_feature.view(-1, all_feature.shape[2])
            all_lable = all_lable.view(-1)
        return all_feature,all_lable

    # 计算对比损失，转化为分类
    def calculate_NCE_loss(self, feat_q, feat_k, labels,namda1=-0.1,namda2=0.1):

        # labels = labels.contiguous().view(-1, 1)
        # if labels.shape[0] != feat_q.shape[0]:
        #     raise ValueError('Num of labels does not match num of features')
        # pos_mask = torch.eq(labels, labels.T).float().to(feat_q.device)
        # neg_mask = 1. - pos_mask

        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        batch_dim_for_bmm = 1

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        
        # 加权重， 对不同类的特征拉远，同类的特征慢慢拉远
        # namda1 = -1
        # namda2 = 0.1
        # l_neg_curbatch = l_neg_curbatch + namda1*pos_mask
        # l_neg_curbatch = l_neg_curbatch + namda2*neg_mask

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=torch.bool)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T

        loss = self.sloss1(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        return loss