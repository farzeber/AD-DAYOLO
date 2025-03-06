'''
@Project ：yolov5-master 
@File    ：twoDomainDataLoader.py
@IDE     ：PyCharm 
@Author  ：付卓彬
@Date    ：2023/7/11 17:58 
'''

import os
import torch
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, check_dataset, check_requirements, check_yaml, clean_str,
                           cv2, is_colab, is_kaggle, segments2boxes, xyn2xy, xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first
import math
import torch
from torch.utils.data.sampler import RandomSampler,SequentialSampler
from torch.utils.data.dataset import ConcatDataset
from utils.fzbdataloaders import LoadImagesAndLabels, InfiniteDataLoader,LoadTwoImagesAndSameLabels
from itertools import chain

def create_domain_dataloader(spath,
                         tpath,
                      imgsz,
                      batch_size,
                      stride,
                      single_cls=False,
                      hyp=None,
                      augment=False,
                      cache=False,
                      pad=0.0,
                      rect=False,
                      rank=-1,
                      workers=8,
                      image_weights=False,
                      quad=False,
                      prefix='',
                      shuffle=False):
    assert batch_size %2 ==0, 'batch_size is not dividived by 2'

    if rect and shuffle:
        LOGGER.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        sdataset = LoadImagesAndLabels(
            spath,
            imgsz,
            batch_size // 2,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix)

        tdataset = LoadImagesAndLabels(
            tpath,
            imgsz,
            batch_size // 2,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix)

    concat_dataset = ConcatDataset([sdataset, tdataset])
    batch_size = min(batch_size, len(concat_dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers

    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates

    sampler = BatchSchedulerSampler(dataset=concat_dataset, batch_size=batch_size // 2)
    return loader(concat_dataset,
                  batch_size=batch_size,
                  shuffle=False,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=True,
                  collate_fn=LoadImagesAndLabels.collate_fn4 if quad else collate_fn), sdataset

def create_threeDomain_dataloader(spath,
                                  stpath,
                         tpath,
                      imgsz,
                      batch_size,
                      stride,
                      single_cls=False,
                      hyp=None,
                      augment=False,
                      cache=False,
                      pad=0.0,
                      rect=False,
                      rank=-1,
                      workers=8,
                      image_weights=False,
                      quad=False,
                      prefix='',
                      shuffle=False):
    assert batch_size %3 ==0, 'batch_size is not dividived by 3'

    if rect and shuffle:
        LOGGER.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        sdataset = LoadTwoImagesAndSameLabels(
            spath,
            stpath,
            imgsz,
            batch_size // 3,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix)

        tdataset = LoadImagesAndLabels(
            tpath,
            imgsz,
            batch_size // 3,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix)

    concat_dataset = ConcatDataset([sdataset, tdataset])
    batch_size = min(batch_size, len(concat_dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers

    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates

    sampler = BatchSchedulerSampler(dataset=concat_dataset, batch_size=batch_size // 3)
    return loader(concat_dataset,
                  batch_size=2 *(batch_size // 3),
                  shuffle=False,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=True,
                  collate_fn=LoadImagesAndLabels.collate_fn4 if quad else collate_three_fn), sdataset

# 加入域标签
def collate_fn(batch):
    im, label, path, shapes, imprior = zip(*batch)  # transposed
    for i, lb in enumerate(label):
        lb[:, 0] = i  # add target image index for build_targets()
    size = len(label) // 2
    return torch.stack(im, 0), torch.cat(label, 0), path, shapes, [0 for x in range(size)] + [1 for x in range(size)], torch.stack(imprior, 0)

def collate_three_fn(batch):
    im, label, path, shapes, imprior = zip(*batch)  # transposed

    labels = []  #加入
    ims = []
    imsP = []
    paths = []
    shapess = []
    #
    for i, lb in enumerate(label):
        labels.append(lb)
        shapess.append(shapes[i])
        if isinstance(im[i], list):
            tempLb = lb.clone()
            labels.append(tempLb)
            tempShape = shapes[i]
            shapess.append(tempShape)
    #
    for i, lb in enumerate(labels):
        lb[:, 0] = i  # add target image index for build_targets()

    for imvalue,pathvalue,impriorvalue in zip(im,path, imprior):
        if isinstance(imvalue, list):
            ims = ims + imvalue
            paths = paths + pathvalue
            assert isinstance(impriorvalue, list), '先验光照同步list失败'
            imsP = imsP + impriorvalue
        else:
            ims.append(imvalue)
            paths.append(pathvalue)
            imsP.append(impriorvalue)
    size = len(labels) // 3
    return torch.stack(ims, 0), torch.cat(labels, 0), paths, shapess, [0 if x % 2 ==0 else 3 for x in range(size + size)] + [1 for x in range(size)],torch.stack(imsP, 0)

def collate_fn_kind(batch):
    im, label, path, shapes, imprior = zip(*batch)  # transposed
    for i, lb in enumerate(label):
        lb[:, 0] = i  # add target image index for build_targets()
    size = len(label) // 2
    return torch.stack(im, 0), torch.cat(label, 0), path, shapes, [0 for x in range(size)] + [1 for x in range(size)]

class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets) # 所有读取的datasets长度总和
        # self.largest_dataset_size = max([len(cur_dataset.samples) for cur_dataset in dataset.datasets])
        self.largest_dataset_size = max([len(cur_dataset) for cur_dataset in dataset.datasets])

    def __len__(self):
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = SequentialSampler(cur_dataset) # 先对每个数据集的iterator进行shuffle
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1] # 找到每个dataset第一个数据的index
        step = self.batch_size * self.number_of_datasets # 步长为每个dataset的mini_batch_size之和
        samples_to_grab = self.batch_size # 每个dataset的mini_batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets # sample数量小的dataset会循环提取

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration: # 把小dataset的数据重复读取，扩充小dataset长度和最大的dataset一样
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)