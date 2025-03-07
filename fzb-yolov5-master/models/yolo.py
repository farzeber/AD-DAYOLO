# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import os
import platform
import sys
import torch
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)
from models.fzbLayers import DANLayer,DANLayer_Clip
from models.HRImageDaLayer import HRDANLayer


from models.integrateDomainLayer import IntegDANLayer
from models.integQKVidaLayer import IntegQKVDANLayer,HrIntegQKVDANLayer,PriorIntegQKVDANLayer,CSWALayer
from models.contrastDomainLayer import contraDANLayer
from models.multiLabelLayer import FEHancer,FEHclassifier,inputMapOut
from models.priorENlayer import RFRB,ImagePENs,IntegImagePENs,ImagePENsAndImageDoamin
from models.DDAYOLOLayers import DDAYOLOMultiLabelClassfier,DDAYOLOImageDA,DDAInstanceDC
from models.dEDALayer import DEDALayer
from models.zeroDceLayer import zeroDceEnhLayer
from models.qinCinLayer import  OutEqualInput,LAGM,FCM
from models.prototypeInstanceLayers import protoSInst,TinstanceLayer
from models.InstanceLayers import InstanceLayer
from models.ganInstLayers import ganInstLayer,ganInstRtinexLayer
from models.oneToOneInstLayer import oneToOneInstLayer
try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        ch1 = ch[:len(anchors)]
        ch2 = ch[len(anchors):]
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch1)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

        # add by fzb
        self.instLayer = InstanceLayer(self,ch2)
        #self.instLayer = TinstanceLayer(self)

        self.twoOutFlag = False

    def forward(self, x):

        # add by fzb begin
        nlength = self.nl
        x1 = x[:nlength]   # detect inference
        x2 = x[nlength:]   # features
        x = x1
        # add by fzb end

        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        #return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
        if x2 and self.training:
            self.twoOutFlag = True
            instanceLabelPre = self.instLayer(x2,[it.detach() for it in x])
            # instanceLabelPre = instanceLabelPre.permute(1,0,2,3)
            # return (x,instanceLabelPre) if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
            return (x,instanceLabelPre)
        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)


    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid(y, x, indexing='ij')
        else:
            yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Model(nn.Module):
    # YOLOv5 model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save , self.outNum= parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        # 添加一个自己的层进去 stride 和 anchors
        if isinstance(m, Detect):
            s = 640  # 2x min stride
            #s = 256  # 2x min stride
            #m.inplace = self.inplace
            m.inplace = False
            pre = self.forward(torch.zeros(2, ch, s, s))
            if m.twoOutFlag:
                pre, _ = pre
            m.stride = torch.tensor([s / x.shape[-2] for x in pre])  # forward
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False,callbackDict ={}, domainlabels = 1, target = ()):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize,callbackDict=callbackDict, domainlabels =domainlabels, target = target)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward   得到第一个detect的值
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False, callbackDict ={}, domainlabels = 1, target = ()):
        y, dt = [], []  # outputs
        dmainThree = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            #if type(m) is RFRB:DEDALayer
            if (type(m) is DEDALayer) or (type(m) is zeroDceEnhLayer):
                x = m(x, domainlabels)
                callbackDict['DEDALayer'] = x.detach()  # 不用detach可能导致显存溢出
            elif type(m) is protoSInst:
                x = m(x,target)
                callbackDict['protoSInstLayer'] = m
            elif type(m) is IntegDANLayer:
                x = m(x,domainlabels)
            elif (type(m) is ganInstLayer) or (type(m) is ganInstRtinexLayer) or (type(m) is oneToOneInstLayer):
                x = m(x, target, domainlabels)
                callbackDict['ganInstLayer'] = x
            else:
                x = m(x)  # run    这里得要改代码，用于目标域的图像恢复

            y.append(x if m.i in self.save else None)  # save output
            # add by fzb for inter output
            if (type(m) is IntegDANLayer) or (type(m) is IntegQKVDANLayer) or (type(m) is HrIntegQKVDANLayer) or (type(m) is PriorIntegQKVDANLayer) or (type(m) is CSWALayer):
                callbackDict['IntegDANLayer'] = x  # add by fzb  DDAYOLOMultiLabelClassfier
            elif type(m) is FEHclassifier:
                callbackDict['FEHclassifier'] = x  # add by fzb
            elif type(m) is DDAYOLOMultiLabelClassfier:
                callbackDict['DDAYOLOMultiLabelClassfier'] = x  # add by fzb
            elif type(m) is DDAInstanceDC:
                callbackDict['DDAInstanceDC'] = x  # add by fzb
            elif (type(m) is DANLayer) or (type(m) is HRDANLayer) or (type(m) is DANLayer_Clip):
                dmainThree.append(x)
            elif type(m) is contraDANLayer:
                dmainThree.append(x)
            elif (type(m) is ImagePENs) or (type(m) is DDAYOLOImageDA):   # DDAYOLOImageDA
                callbackDict['ImagePENs'] = x  # add by fzb
            elif (type(m) is IntegImagePENs) or (type(m) is ImagePENsAndImageDoamin):
                callbackDict['ImagePENs'] = x  # add by fzb
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        callbackDict['hiddenout'] = dmainThree  # add by fzb
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1).detach()  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


def parse_model(d, ch):  # model_dict, input_channels(3)
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    out = []           # add by fzb
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args  考虑添加flag用来输出
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in (Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x):
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost, C3x]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        elif m is DANLayer or (m is DANLayer_Clip):              # add by fzb
            c1 = ch[f]
            args = [c1,*args]
            out.append(i)
        elif (m is IntegDANLayer) or (m is IntegQKVDANLayer) or (m is HrIntegQKVDANLayer) or (m is PriorIntegQKVDANLayer) or (m is CSWALayer):              # add by fzb
            args.append([ch[x] for x in f])
        elif (m is oneToOneInstLayer):              # add by fzb
            args.append([ch[x] for x in f])
        elif m is FEHancer:              # add by fzb
            args.append([ch[x] for x in f])
            c2 = 3
        elif m is FEHclassifier:              # add by fzb
            args.append(ch[f])
        elif m is contraDANLayer:
            args.append(ch[f])
        elif m is RFRB:
            for x in f:
                args.append(ch[x])
            c2 = ch[f[-1]]
        elif (m is ImagePENs) or (m is DDAYOLOImageDA):
            args.append([ch[x] for x in f])
        elif (m is IntegImagePENs) or (m is ImagePENsAndImageDoamin):
            args.append([ch[x] for x in f])
        elif m is DDAYOLOMultiLabelClassfier:
            args[0] = ch[f]
        elif m is DDAInstanceDC:
            args.append([ch[x] for x in f])
        elif (m is protoSInst) or (m is ganInstLayer):
            args.append([ch[x] for x in f])
        else:
            if type(f) is list:
                c2 = ch[f[0]]
            else:
                c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist    -1 表示上一层
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save), out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        _ = model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()
