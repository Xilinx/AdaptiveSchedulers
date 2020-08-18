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
import time
import logging
import argparse
import random
import numpy as np

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

from utils.utils import report_accuracy, get_composite_metric

from tensorboardX import SummaryWriter
from aslr import ASLR
from utils.resnet import resnet20_cifar10
from utils.cifar10_dataset import CIFAR10MetaInfo, get_train_data_source, get_val_data_source


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Pytoch model for image classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--work-dir",
        type=str,
        default=os.path.join("..", "imgclsmob_data"),
        help="path to working directory only for dataset root path preset")

    args, _ = parser.parse_known_args()
    dataset_metainfo = CIFAR10MetaInfo()
    dataset_metainfo.add_dataset_parser_arguments(
        parser=parser,
        work_dir_path=args.work_dir)

    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="number of gpus to use")
    parser.add_argument(
        "-j",
        "--num-data-workers",
        dest="num_workers",
        default=4,
        type=int,
        help="number of preprocessing workers")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="training batch size per device (CPU/GPU)")
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=120,
        help="number of training epochs")
    parser.add_argument(
        "--optimizer-name",
        type=str,
        default="nag",
        help="optimizer name")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="learning rate")
    parser.add_argument(
        "--lr-mode",
        type=str,
        default="multistep",
        help="learning rate scheduler mode")
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.1,
        help="decay rate of learning rate")
    parser.add_argument(
        "--lr-decay-epoch",
        type=str,
        default="40,60",
        help="epoches at which learning rate decays")
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="momentum value for optimizer")
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0001,
        help="weight decay rate")

    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed to be fixed")

    parser.add_argument(
        "--tensor-dir-path",
        type=str,
        default="",
        help="Tensorboard summary writer saving path")

    args = parser.parse_args()
    return args


def prepare_trainer(net, optimizer_name, wd, momentum, lr_mode, lr, lr_decay_epoch, lr_decay, num_epochs):

    net_layers = separate_layers(net)
    optimizer_name = optimizer_name.lower()
    if (optimizer_name == "sgd") or (optimizer_name == "nag"):
        optimizer = torch.optim.SGD(
            params=net_layers,
            lr=lr,
            momentum=momentum,
            weight_decay=wd,
            nesterov=(optimizer_name == "nag"))
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            params=net_layers,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=wd,
            amsgrad=False
        )
    else:
        raise ValueError("Usupported optimizer: {}".format(optimizer_name))

    cudnn.benchmark = True

    lr_mode = lr_mode.lower()
    lr_decay_epoch = [int(i) for i in lr_decay_epoch.split(",")]
    if lr_mode == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=lr_decay_epoch,
            gamma=lr_decay,
            last_epoch=-1)
    elif lr_mode == 'aslr':
        lr_scheduler = ASLR(optimizer=optimizer)
    elif lr_mode == "cosine":
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=num_epochs,
            last_epoch=(num_epochs - 1))
    else:
        raise ValueError("Usupported lr_scheduler: {}".format(lr_mode))

    return optimizer, lr_scheduler


def train_epoch(epoch, net, train_metric, train_data, use_cuda, L, optimizer, batch_size, writer, args, lr_scheduler):

    tic = time.time()
    net.train()
    train_metric.reset()
    train_loss = 0.0

    btic = time.time()
    for i, (data, target) in enumerate(train_data):
        if use_cuda:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        output = net(data)
        loss = L(output, target)
        optimizer.zero_grad()
        loss.backward()

        # _, thetas, vis = optimizer.step()
        optimizer.step()

        train_loss += loss.item()

        train_metric.update(
            labels=target,
            preds=output)

        niter = epoch * len(train_data) + i
        speed = batch_size / (time.time() - btic)
        btic = time.time()
        train_accuracy_msg = report_accuracy(metric=train_metric)
        writer.add_scalar('Train/Error', float(train_accuracy_msg.split('=')[1]), niter)
        writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]["lr"], niter)
        writer.add_scalar('Train/Loss', loss.item(), niter)
        logging.info("Epoch[{}] Batch [{}]\tSpeed: {:.2f} samples/sec\t{}\tlr={:.5f}".format(
            epoch + 1, i, speed, train_accuracy_msg, optimizer.param_groups[0]["lr"]))

        if args.lr_mode == 'aslr':
            lr_scheduler.step()

    throughput = int(batch_size * (i + 1) / (time.time() - tic))
    logging.info("[Epoch {}] speed: {:.2f} samples/sec\ttime cost: {:.2f} sec".format(
        epoch + 1, throughput, time.time() - tic))

    train_loss /= (i + 1)
    train_accuracy_msg = report_accuracy(metric=train_metric)
    logging.info("[Epoch {}] training: {}\tloss={:.4f}".format(
        epoch + 1, train_accuracy_msg, train_loss))

    train_error = float(train_accuracy_msg.split('=')[1])

    return train_loss, train_error


def train_net(batch_size, num_epochs, train_data, val_data, net, optimizer, lr_scheduler,
              num_classes, val_metric, train_metric, use_cuda, writer, args):

    best_top1 = 0
    assert (num_classes > 0)

    L = nn.CrossEntropyLoss()
    if use_cuda:
        L = L.cuda()

    gtic = time.time()
    for epoch in range(num_epochs):
        lr_scheduler.step()

        train_loss, train_error = train_epoch(
            epoch=epoch,
            net=net,
            train_metric=train_metric,
            train_data=train_data,
            use_cuda=use_cuda,
            L=L,
            optimizer=optimizer,
            batch_size=batch_size,
            writer=writer,
            args=args,
            lr_scheduler=lr_scheduler)

        validate(
            metric=val_metric,
            net=net,
            val_data=val_data,
            use_cuda=use_cuda)
        val_accuracy_msg = report_accuracy(metric=val_metric)
        logging.info("[Epoch {}] validation: {}".format(epoch + 1, val_accuracy_msg))

        val_error = float(val_accuracy_msg.split('=')[1].split(',')[0])
        writer.add_scalar('Validation/Error_epoch', val_error, epoch + 1)

        if args.lr_mode == 'aslr':
            lr_scheduler.update_loss(val_error)

        if best_top1 < 1 - val_error:
            best_top1 = 1 - val_error
        writer.add_scalar('Validation/Best_top1_epoch', best_top1, epoch + 1)

        writer.add_scalar('Train/Error_epoch', train_error, epoch + 1)
        writer.add_scalar('Train/Loss_epoch', train_loss, epoch + 1)

        writer.add_scalar('Difference/Error_val_train', val_error - train_error, epoch + 1)

    logging.info("Total time cost: {:.2f} sec".format(time.time() - gtic))


def separate_layers(net):
    layers = []
    for name, param in net.named_parameters():
        print('name: {}'. format(name))
        layers.append({'params': param})
    return layers


def get_net(use_cuda=False):
    net = resnet20_cifar10()

    if use_cuda:
        net = torch.nn.DataParallel(net)
        net = net.cuda()

    return net


def validate(metric,
             net,
             val_data,
             use_cuda):
    """
    Core validation/testing routine.

    Parameters:
    ----------
    metric : EvalMetric
        Metric object instance.
    net : Module
        Model.
    val_data : DataLoader
        Data loader.
    use_cuda : bool
        Whether to use CUDA.

    Returns
    -------
    EvalMetric
        Metric object instance.
    """
    net.eval()
    metric.reset()
    with torch.no_grad():
        for data, target in val_data:
            if use_cuda:
                target = target.cuda(non_blocking=True)
            output = net(data)
            metric.update(target, output)
    return metric


def main():
    """
    Main body of script.
    """
    args = parse_args()

    if args.seed <= 0:
        args.seed = np.random.randint(10000)
    else:
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    use_cuda = args.num_gpus > 0
    batch_size = max(1, args.num_gpus)*args.batch_size

    num_classes = 10
    net = get_net(use_cuda)

    ds_metainfo = CIFAR10MetaInfo()
    ds_metainfo.update(args=args)

    train_data = get_train_data_source(
        ds_metainfo=ds_metainfo,
        batch_size=batch_size,
        num_workers=args.num_workers)
    val_data = get_val_data_source(
        ds_metainfo=ds_metainfo,
        batch_size=batch_size,
        num_workers=args.num_workers)

    optimizer, lr_scheduler = prepare_trainer(
        net=net,
        optimizer_name=args.optimizer_name,
        wd=args.wd,
        momentum=args.momentum,
        lr_mode=args.lr_mode,
        lr=args.lr,
        lr_decay_epoch=args.lr_decay_epoch,
        lr_decay=args.lr_decay,
        num_epochs=args.num_epochs)

    _writer = SummaryWriter(args.tensor_dir_path)

    train_net(
        batch_size=batch_size,
        num_epochs=args.num_epochs,
        train_data=train_data,
        val_data=val_data,
        net=net,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        num_classes=num_classes,
        val_metric=get_composite_metric(ds_metainfo.val_metric_names, ds_metainfo.val_metric_extra_kwargs),
        train_metric=get_composite_metric(ds_metainfo.train_metric_names, ds_metainfo.train_metric_extra_kwargs),
        use_cuda=use_cuda,
        writer=_writer,
        args=args)


if __name__ == "__main__":
    main()
