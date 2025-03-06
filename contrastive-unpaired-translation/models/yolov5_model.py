'''
@Project ：contrastive-unpaired-translation 
@File    ：yolov5_model.py
@IDE     ：PyCharm 
@Author  ：付卓彬
@Date    ：2024/6/6 17:23 
'''
from .yolo import Model
from .yolo_Config import *
import torch
from pathlib import Path
import yaml
dir = Path(__file__).parent

def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def getYoloModel(nc,device,imgsz=286, cfg = dir / 'yolov5s.yaml', weights = 'best.pt', hypfile = dir/'hyp.scratch-low.yaml'):
    weights = dir / weights
    with open(hypfile, encoding='ascii', errors='ignore') as f:
        hyp = yaml.safe_load(f)  # model dict

        model = Model(cfg, ch=3, nc=nc).to(device)  # create
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict())  # intersect
        model.load_state_dict(csd, strict=True)  # load
        model.eval()

        nl = de_parallel(model).model[-1].nl
        hyp['box'] *= 3 / nl  # scale to layers
        hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
        hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        model.hyp = hyp

        set_requires_grad(model, False)
        return model

if __name__ == "__main__":
    if torch.cuda.is_available():
        arg = 'cuda:0'
        device = torch.device(arg)
        getYoloModel(5, device)