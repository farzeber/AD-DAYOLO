'''
@Project ：contrastive-unpaired-translation 
@File    ：yolov5_model.py
@IDE     ：PyCharm 
@Author  ：付卓彬
@Date    ：2024/6/6 17:23 
'''
from yolo import Model
import torch
from pathlib import Path

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

def getYoloModel(nc,device, cfg = dir / 'yolov5s.yaml', weights = dir/ 'best.pt'):

    model = Model(cfg, ch=3, nc=nc).to(device)  # create
    ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
    csd = (ckpt.get('ema') or ckpt['model']).float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict())  # intersect

    model.load_state_dict(csd, strict=True)  # load
    model.eval()
    set_requires_grad(model, False)
    return model

if torch.cuda.is_available():
    arg = 'cuda:0'
    device = torch.device(arg)
    model = getYoloModel(5, device)
    print('get')