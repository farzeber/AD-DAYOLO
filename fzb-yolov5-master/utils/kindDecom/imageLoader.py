'''
@Project ：kindFuxian 
@File    ：imageLoader.py
@IDE     ：PyCharm 
@Author  ：付卓彬
@Date    ：2023/3/13 15:38 
'''


#==========================================step 1/5 准备数据===============================
from torchvision import transforms
import torchvision.transforms.functional as transFuc
from torch.utils.data import Dataset
import os
from PIL import Image
from tqdm import tqdm
import torch
import random
from glob import glob

#在这里得到图片的patch，然后
class TrainDataset(Dataset):
    def __init__(self, low_dir,high_dir, patch_size):
        if not patch_size:
            self.patch_size = 48
        else:
            self.patch_size = patch_size
        low_img_names = os.listdir(low_dir)
        low_img_names = list(filter(lambda x: x.endswith('.png'), low_img_names))
        low_img_names.sort(key=lambda x: int(x[:-4]))
        high_img_names = os.listdir(high_dir)
        high_img_names = list(filter(lambda x: x.endswith('.png'), high_img_names))
        high_img_names.sort(key=lambda x: int(x[:-4]))
        assert len(low_img_names) == len(high_img_names)
        low_Pic = []
        high_Pic = []
        for path1,path2 in tqdm(zip(low_img_names,high_img_names)):
            img = Image.open(os.path.join(low_dir,path1))
            img_tensor = transforms.ToTensor()(img)
            min = torch.min(img_tensor)
            max = torch.max(img_tensor)
            img_tensor = torch.div(img_tensor - min, torch.Tensor(torch.maximum((max - min), torch.Tensor([0.001]))))
            low_Pic.append(img_tensor)

            img = Image.open(os.path.join(high_dir,path2))
            img_tensor = transforms.ToTensor()(img)
            min = torch.min(img_tensor)
            max = torch.max(img_tensor)
            img_tensor = torch.div(img_tensor - min, torch.Tensor(torch.maximum((max - min), torch.Tensor([0.001]))))
            high_Pic.append(img_tensor)
        assert len(low_Pic) == len(high_Pic)
        self.lowPic = low_Pic
        self.highPic = high_Pic
        self.data_info = list(zip(low_Pic,high_Pic))

    def __getitem__(self, index):
        patch_size = self.patch_size
        low_img, heigh_image = self.data_info[index]
        # 应用变换
        _,h, w = low_img.shape
        x = random.randint(0, h - self.patch_size)
        y = random.randint(0, w - self.patch_size)
        low_img=low_img[:,x : x+patch_size, y : y+patch_size]
        heigh_image = heigh_image[:,x: x + patch_size,y: y + patch_size]
        mode = random.randint(0, 7)
        # 数据增强
        low_img = self.data_augmentation(low_img,mode)
        heigh_image = self.data_augmentation(heigh_image,mode)
        return (low_img, heigh_image)

    @staticmethod
    def data_augmentation(image, mode):
        if mode == 0:
            # original
            return image
        elif mode == 1:
            # flip up and down
            return torch.flip(image,[1])
        elif mode == 2:
            # rotate counterwise 90 degree
            return torch.rot90(image,k=1,dims=[1,2])
        elif mode == 3:
            # rotate 90 degree and flip up and down
            image = torch.rot90(image,k=1,dims=[1,2])
            return torch.flip(image,[1])
        elif mode == 4:
            # rotate 180 degree
            return torch.rot90(image, k=2,dims=[1,2])
        elif mode == 5:
            # rotate 180 degree and flip
            image = torch.rot90(image, k=2, dims=[1,2])
            return torch.flip(image,dims=[1])
        elif mode == 6:
            # rotate 270 degree
            return torch.rot90(image, k=3,dims=[1,2])
        elif mode == 7:
            # rotate 270 degree and flip
            image = torch.rot90(image, k=3, dims=[1,2])
            return torch.flip(image,dims=[1])

    def __len__(self):
        return len(self.data_info)


class EvalDataset(Dataset):
    def __init__(self,eval_low_dir, eval_high_dir):
        eval_low_data_name = glob(eval_low_dir)
        eval_low_data_name.sort()
        eval_high_data_name = glob(eval_high_dir)
        eval_high_data_name.sort()
        assert len(eval_low_data_name) == len(eval_high_data_name)
        eval_lowPic = []
        eval_highPic = []
        for lowPath,highPath in tqdm(zip(eval_low_data_name,eval_high_data_name)):
            lowPic = Image.open(lowPath)
            highPic = Image.open(highPath)
            low_tensor = transFuc.to_tensor(lowPic)
            high_tensor = transFuc.to_tensor(highPic)

            min_low = torch.min(low_tensor)
            max_low = torch.max(low_tensor)
            low_tensor = torch.div(low_tensor - min_low, torch.maximum((max_low - min_low), torch.Tensor([0.001])))
            eval_lowPic.append(low_tensor)

            min_high = torch.min(high_tensor)
            max_high = torch.max(high_tensor)
            high_tensor = torch.div(high_tensor - min_high, torch.maximum((max_high - min_high), torch.Tensor([0.001])))
            eval_highPic.append(high_tensor)
        assert len(eval_lowPic) == len(eval_highPic)
        self.data = list(zip(eval_lowPic, eval_highPic))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


