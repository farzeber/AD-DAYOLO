'''
@Project ：yolov5-master 
@File    ：fzbutils.py
@IDE     ：PyCharm 
@Author  ：付卓彬
@Date    ：2023/8/25 16:10 
'''
import random

import numpy as np
import torch
from PIL import Image
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
def saveFakePic(imgs,cat_image, filepath, save_dir, epoch, i):
    # imgs,interPicture, paths, save_dir, epoch
    with torch.no_grad():
        length = imgs.shape[0]
        indic = random.randint(0, length -1)
        #indic = random.randint(length //2, length - 1)
        filename = os.path.basename(filepath[indic])
        filename_without_ext, file_extension = os.path.splitext(filename)

        imgs = imgs[indic]
        imgs = imgs.permute(1, 2, 0)

        cat_image = cat_image[indic]
        cat_image = cat_image.permute(1,2,0)
        shape = cat_image.shape

        outimage = torch.zeros_like(imgs)
        outimage = outimage.repeat(2,1,1)
        outimage[:shape[0],:shape[1],...] = cat_image
        outimage[shape[0]:(shape[0]+imgs.shape[0]), ...] = imgs

        outimage = outimage.cpu().float().numpy()
        im = Image.fromarray(np.clip(outimage * 255.0, 0, 255.0).astype('uint8'))
        if save_dir is None:
            save_dir = ROOT
        im.save(save_dir / f'{epoch}_{i}_{filename_without_ext}.png', 'png')
