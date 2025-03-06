'''
@Project ：kindFuxian 
@File    ：kindutil.py
@IDE     ：PyCharm 
@Author  ：付卓彬
@Date    ：2023/3/14 15:10 
'''

import torch
import torchvision.transforms.functional as transFuc
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F

def save_images(filepath, result_1, result_2 = None, fzborigin = None):
    #光照图像是一通道的
    result_2 = torch.cat([result_2, result_2, result_2], dim=0)
    result_1 = torch.squeeze(result_1)
    result_2 = torch.squeeze(result_2)
    fzborigin = torch.squeeze(fzborigin)
    rebuildPic = result_1 * result_2
    if not result_2.any():
        cat_image = result_1
    else:
        cat_image = torch.cat([result_1, result_2], dim=2)
    if not fzborigin.any():
        cat_image = cat_image
    else:
        rebuildPic1 = torch.cat([rebuildPic, fzborigin], dim=2)
        cat_image = torch.cat([cat_image, rebuildPic1], dim=1)
    #imageOri = torch.clip(cat_image * 255, 0, 255)
    image = transFuc.to_pil_image(cat_image)
    #image.show()
    image.save(filepath, 'png')
    #重建图片

    fzborigin = torch.permute(fzborigin,(1,2,0))
    rebuildPic = torch.permute(rebuildPic, (1,2,0))
    fzboriginCpu = fzborigin.cpu().numpy()
    rebuildPicCpu = rebuildPic.cpu().detach().numpy()
    psnrValue = psnr(fzboriginCpu,rebuildPicCpu)
    ssimValue = ssim(fzboriginCpu, rebuildPicCpu, multichannel=True)

    print("psnr==%f sim==%f" % (psnrValue, ssimValue))


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

def save_images_two(filepath, lowPic, enhancePic,highPic):
    #光照图像是一通道的

    result_1 = torch.squeeze(lowPic)
    result_2 = torch.squeeze(enhancePic)
    result_3 = torch.squeeze(highPic)
    if not result_2.any():
        cat_image = result_1
    else:
        cat_image = torch.cat([result_1, result_2], dim=2)

    if not result_3.any():
        pass
    else:
        cat_image = torch.cat([cat_image, result_3], dim=2)
    #imageOri = torch.clip(cat_image * 255, 0, 255)
    image = transFuc.to_pil_image(cat_image)
    #image.show()
    image.save(filepath, 'png')
    #重建图片

    fzborigin = torch.permute(result_3,(1,2,0))
    rebuildPic = torch.permute(result_2, (1,2,0))
    fzboriginCpu = fzborigin.cpu().numpy()
    rebuildPicCpu = rebuildPic.cpu().detach().numpy()
    psnrValue = psnr(fzboriginCpu,rebuildPicCpu)
    ssimValue = ssim(fzboriginCpu, rebuildPicCpu, multichannel=True)

    print("psnr==%f sim==%f" % (psnrValue, ssimValue))