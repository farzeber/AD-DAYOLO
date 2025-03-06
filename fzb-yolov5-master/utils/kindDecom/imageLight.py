'''
调用各分层训练结构
加载数据集
'''
from utils.kindDecom.decom import DecomLayer
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torchvision import transforms
from utils.kindDecom.imageLoader import TrainDataset,EvalDataset
from torch.utils.data.dataloader import DataLoader
import time


def gradient(input_tensor, direction, device = None):
    kernal_x = torch.tensor([0.,0.,-1.,1.])
    b = torch.zeros(1)
    if device:
        kernal_x = kernal_x.to(device)
        b = b.to(device)
    kernal_x = torch.reshape(kernal_x,(1,1,2,2))

    kernal_y = torch.permute(kernal_x,(0,1,3,2))
    if direction == "x":
        kernel = kernal_x
    elif direction == "y":
        kernel = kernal_y
    gradient_orig = F.conv2d(input_tensor, kernel,bias=b,padding='same')
    gradient_abs = torch.abs(gradient_orig)
    grad_min = torch.min(gradient_abs)
    grad_max = torch.max(gradient_abs)
    grad_norm = torch.div((gradient_abs - grad_min),(grad_max - grad_min + 0.0001))
    return grad_norm

#相互光照
def mutual_i_loss(input_I_low, input_I_high,device):
    low_gradient_x = gradient(input_I_low, "x",device)
    high_gradient_x = gradient(input_I_high, "x",device)
    x_loss = (low_gradient_x + high_gradient_x)* torch.exp(-10*(low_gradient_x+high_gradient_x))
    low_gradient_y = gradient(input_I_low, "y",device)
    high_gradient_y = gradient(input_I_high, "y",device)
    y_loss = (low_gradient_y + high_gradient_y) * torch.exp(-10*(low_gradient_y+high_gradient_y))
    mutual_loss = torch.mean( x_loss + y_loss)
    return mutual_loss

#光照平滑
def mutual_i_input_loss(input_I_low, input_im, device):
    input_gray = transforms.Grayscale(num_output_channels=1)(input_im)
    low_gradient_x = gradient(input_I_low, "x",device)
    input_gradient_x = gradient(input_gray, "x", device)
    c = torch.Tensor([0.01])
    c = c.to(device)
    x_loss = torch.abs(torch.div(low_gradient_x, torch.maximum(input_gradient_x, c )))
    low_gradient_y = gradient(input_I_low, "y", device)
    input_gradient_y = gradient(input_gray, "y", device)
    y_loss = torch.abs(torch.div(low_gradient_y, torch.maximum(input_gradient_y, c)))
    mut_loss = torch.mean(x_loss + y_loss)
    return mut_loss


class My_loss(nn.Module):
    def __init__(self,device):
        super().__init__()  # 没有需要保存的参数和状态信息
        self.device = device

    def forward(self, R_low, I_low,R_high,I_high,img_batch,label_batch):  # 定义前向的函数运算即可
        I_low_3 = torch.concat([I_low, I_low, I_low], axis=1)
        I_high_3 = torch.concat([I_high, I_high, I_high], axis=1)
        # 重建损失
        recon_loss_low = torch.mean(torch.abs(R_low * I_low_3 - img_batch))
        recon_loss_high = torch.mean(torch.abs(R_high * I_high_3 - label_batch))
        # 分解反射损失
        equal_R_loss = torch.mean(torch.abs(R_low - R_high))
        # 相互光照损失  强化边缘信息
        i_mutual_loss = mutual_i_loss(I_low, I_high, self.device)
        # 光照平滑
        i_input_mutual_loss_high = mutual_i_input_loss(I_high, label_batch, self.device)
        i_input_mutual_loss_low = mutual_i_input_loss(I_low, img_batch, self.device)
        loss_Decom = 1 * recon_loss_high + 1 * recon_loss_low \
                     + 0.01 * equal_R_loss + 0.2 * i_mutual_loss \
                     + 0.15 * i_input_mutual_loss_high + 0.15 * i_input_mutual_loss_low
        return loss_Decom

#加载模型

class KindModel():
    def __init__(self, device):
        super().__init__()
        model_save_dir = r'./utils/kindDecom/decomModel/'
        # /public/home/jd_fzb/workspace/yolo_prior_kinddecom/utils/kindDecom/decomModel
        # D:\paper\codes\yolov5-master\yolov5-master\utils\kindDecom
        if not os.path.isdir(model_save_dir):
            os.makedirs(model_save_dir)
        decomLayer = DecomLayer(3)
        models = os.listdir(model_save_dir)
        if models:
            models = list(filter(lambda x: x.endswith('.ptx'), models))
            models = sorted(models, key=lambda x: os.path.getctime(os.path.join(model_save_dir, x)))
            if len(models) > 0:
                modelPath = os.path.join(model_save_dir, models[0])
                print("加载KindDecom预训练模型：%s" % modelPath)
                decomLayer.load_state_dict(torch.load(modelPath))
                # if torch.cuda.is_available():
                #     print("KindDecom预训练模型使用gpu训练")
                #     device = torch.device("cuda")
        decomLayer = decomLayer.to(device)
        decomLayer.eval()
        self.model = decomLayer

    def getImageLight(self,src):
        decomLayer = self.model
        decomLayer.eval()
        src = src.clone()
        with torch.no_grad():
            R_image, I_image = decomLayer(src)
            dst = torch.real(I_image)
            dst = torch.clip(I_image*255, 0, 255)
            dst = dst.type(torch.uint8)
        return dst

if __name__ == "__main__":
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("使用gpu训练")
        device = torch.device("cuda")

    sample_dir = './Decom_net_train/'
    if not os.path.isdir(sample_dir):
        os.makedirs(sample_dir)

    #模型
    model_save_dir = './ModelSaverDir/'
    if not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir)

    batch_size = 10
    patch_size = 48
    epochNum = 2000
    learning_rate = 0.0001
    eval_every_epoch = 200
    lowPath = './LOLdataset/our485/low'
    highPath = "./LOLdataset/our485/high"

    # 测试数据加载
    evalDataSet = EvalDataset('./LOLdataset/eval15/low/*.png*','./LOLdataset/eval15/high/*.png*')
    evalDataLoader = DataLoader(dataset=evalDataSet, batch_size=1)
    # 训练数据加载
    dataset = TrainDataset(lowPath, highPath, patch_size)
    numBatch = len(dataset) // int(batch_size)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
    #可以进行模型加载    按什么排列
    decomLayer = DecomLayer(3)
    models = os.listdir(model_save_dir)
    if models:
        models = list(filter(lambda x: x.endswith('.ptx'), models))
        models = sorted(models, key=lambda x: os.path.getctime(os.path.join(model_save_dir, x)))
        if len(models) > 0:
            modelPath = os.path.join(model_save_dir, models[0])
            print("加载预训练模型：%s" % modelPath)
            decomLayer.load_state_dict(torch.load(modelPath))
    decomLayer = decomLayer.to(device)
    optm = torch.optim.Adam(decomLayer.parameters(), lr=learning_rate)
    lossF = My_loss(device)
    lossF = lossF.to(device)

    start_time = time.time()
    for epoth in range(0, epochNum):
        index = 0
        decomLayer.train()
        for img_batch, label_batch in dataloader:
            img_batch = img_batch.to(device)
            label_batch = label_batch.to(device)
            R_low, I_low = decomLayer(img_batch)
            R_high, I_high = decomLayer(label_batch)
            loss = lossF(R_low, I_low,R_high,I_high,img_batch,label_batch)
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                  % (epoth + 1, index + 1, numBatch, time.time() - start_time, loss))
            optm.zero_grad()
            loss.backward()
            optm.step()
            index = index + 1
        if (epoth + 1) % eval_every_epoch == 0:
            # 进行eval评估 同时进行模型保存
            decomLayer.eval()
            print("eval.........epoth = ",epoth)
            for idx,v in zip(range(len(evalDataLoader)),evalDataLoader):
                #图像变成batch
                lowPic, highPic = v
                lowPic = lowPic.to(device)
                highPic = highPic.to(device)
                R_low, I_low = decomLayer(lowPic)
                # save_images(os.path.join(sample_dir, 'low_%d_%d.png' % ( idx + 1, epoth + 1)), R_low, I_low, fzborigin=lowPic)
                R_high, I_high = decomLayer(highPic)
                # save_images(os.path.join(sample_dir, 'high_%d_%d.png' % (idx + 1, epoth + 1)), R_high, I_high, fzborigin=highPic)

            # 模型保存 命名日期加 epoth  需要变成cpu吗？
            torch.save(decomLayer.state_dict(), os.path.join(model_save_dir,"decom_%s_epoch%d.ptx" % (str(time.strftime('%Y%m%d')), epoth)))
    print("wrong")

