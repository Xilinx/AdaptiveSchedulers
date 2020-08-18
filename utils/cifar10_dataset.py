###############################################################################
#  Copyright (c) 2019-2020, Xilinx, Inc.
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1.  Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#
#  2.  Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#  3.  Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
#  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
#  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
#  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
#  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################

###############################################################################
# Author: Alireza Khodamoradi
###############################################################################

import os
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


class DatasetMetaInfo(object):

    def __init__(self):
        self.use_imgrec = False
        self.label = None
        self.root_dir_name = None
        self.root_dir_path = None
        self.dataset_class = None
        self.dataset_class_extra_kwargs = None
        self.num_training_samples = None
        self.in_channels = None
        self.num_classes = None
        self.input_image_size = None
        self.train_metric_capts = None
        self.train_metric_names = None
        self.train_metric_extra_kwargs = None
        self.train_use_weighted_sampler = False
        self.val_metric_capts = None
        self.val_metric_names = None
        self.val_metric_extra_kwargs = None
        self.test_metric_capts = None
        self.test_metric_names = None
        self.test_metric_extra_kwargs = None
        self.saver_acc_ind = None
        self.ml_type = None
        self.allow_hybridize = True
        self.net_extra_kwargs = None
        self.load_ignore_extra = False

    def add_dataset_parser_arguments(self,
                                     parser,
                                     work_dir_path):
        parser.add_argument(
            "--data-dir",
            type=str,
            default=os.path.join(work_dir_path, self.root_dir_name),
            help="path to directory with {} dataset".format(self.label))
        parser.add_argument(
            "--num-classes",
            type=int,
            default=self.num_classes,
            help="number of classes")
        parser.add_argument(
            "--in-channels",
            type=int,
            default=self.in_channels,
            help="number of input channels")

    def update(self,
               args):
        self.root_dir_path = args.data_dir
        self.num_classes = args.num_classes
        self.in_channels = args.in_channels


class CIFAR10Wrapper(CIFAR10):
    def __init__(self,
                 root=os.path.join("~", ".torch", "datasets", "cifar10"),
                 mode="train",
                 transform=None):
        super(CIFAR10Wrapper, self).__init__(
            root=root,
            train=(mode == "train"),
            transform=transform,
            download=True)


class CIFAR10MetaInfo(DatasetMetaInfo):
    def __init__(self):
        super(CIFAR10MetaInfo, self).__init__()
        self.label = "CIFAR10"
        self.short_label = "cifar"
        self.root_dir_name = "cifar10"
        self.dataset_class = CIFAR10Wrapper
        self.num_training_samples = 50000
        self.in_channels = 3
        self.num_classes = 10
        self.input_image_size = (32, 32)
        self.train_metric_capts = ["Train.Err"]
        self.train_metric_names = ["Top1Error"]
        self.train_metric_extra_kwargs = [{"name": "err"}]
        self.val_metric_capts = ["Val.Err"]
        self.val_metric_names = ["Top1Error"]
        self.val_metric_extra_kwargs = [{"name": "err"}]
        self.saver_acc_ind = 0
        self.train_transform = cifar10_train_transform
        self.val_transform = cifar10_val_transform
        self.test_transform = cifar10_val_transform
        self.ml_type = "imgcls"


def cifar10_train_transform(ds_metainfo,
                            mean_rgb=(0.4914, 0.4822, 0.4465),
                            std_rgb=(0.2023, 0.1994, 0.2010),
                            jitter_param=0.4):
    assert (ds_metainfo is not None)
    assert (ds_metainfo.input_image_size[0] == 32)
    return transforms.Compose([
        transforms.RandomCrop(
            size=32,
            padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=jitter_param,
            contrast=jitter_param,
            saturation=jitter_param),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb)
    ])


def cifar10_val_transform(ds_metainfo,
                          mean_rgb=(0.4914, 0.4822, 0.4465),
                          std_rgb=(0.2023, 0.1994, 0.2010)):
    assert (ds_metainfo is not None)
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb)
    ])


def get_train_data_source(ds_metainfo, batch_size, num_workers):

    transform_train = ds_metainfo.train_transform(ds_metainfo=ds_metainfo)
    kwargs = ds_metainfo.dataset_class_extra_kwargs if ds_metainfo.dataset_class_extra_kwargs is not None else {}
    dataset = ds_metainfo.dataset_class(
        root=ds_metainfo.root_dir_path,
        mode="train",
        transform=transform_train,
        **kwargs)
    if not ds_metainfo.train_use_weighted_sampler:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True)
    else:
        sampler = WeightedRandomSampler(
            weights=dataset.sample_weights,
            num_samples=len(dataset))
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            # shuffle=True,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True)


def get_val_data_source(ds_metainfo, batch_size, num_workers):

    transform_val = ds_metainfo.val_transform(ds_metainfo=ds_metainfo)
    kwargs = ds_metainfo.dataset_class_extra_kwargs if ds_metainfo.dataset_class_extra_kwargs is not None else {}
    dataset = ds_metainfo.dataset_class(
        root=ds_metainfo.root_dir_path,
        mode="val",
        transform=transform_val,
        **kwargs)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)