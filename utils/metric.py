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

from collections import OrderedDict
import numpy as np
import torch


def check_label_shapes(labels, preds, shape=False):

    if not shape:
        label_shape, pred_shape = len(labels), len(preds)
    else:
        label_shape, pred_shape = labels.shape, preds.shape

    if label_shape != pred_shape:
        raise ValueError("Shape of labels {} does not match shape of predictions {}".format(label_shape, pred_shape))


class EvalMetric(object):

    def __init__(self,
                 name,
                 output_names=None,
                 label_names=None,
                 **kwargs):
        super(EvalMetric, self).__init__()
        self.name = str(name)
        self.output_names = output_names
        self.label_names = label_names
        self._has_global_stats = kwargs.pop("has_global_stats", False)
        self._kwargs = kwargs
        self.reset()

    def __str__(self):
        return "EvalMetric: {}".format(dict(self.get_name_value()))

    def get_config(self):

        config = self._kwargs.copy()
        config.update({
            "metric": self.__class__.__name__,
            "name": self.name,
            "output_names": self.output_names,
            "label_names": self.label_names})
        return config

    def update_dict(self, label, pred):

        if self.output_names is not None:
            pred = [pred[name] for name in self.output_names]
        else:
            pred = list(pred.values())

        if self.label_names is not None:
            label = [label[name] for name in self.label_names]
        else:
            label = list(label.values())

        self.update(label, pred)

    def update(self, labels, preds):

        raise NotImplementedError()

    def reset(self):

        self.num_inst = 0
        self.sum_metric = 0.0
        self.global_num_inst = 0
        self.global_sum_metric = 0.0

    def reset_local(self):

        self.num_inst = 0
        self.sum_metric = 0.0

    def get(self):

        if self.num_inst == 0:
            return self.name, float("nan")
        else:
            return self.name, self.sum_metric / self.num_inst

    def get_global(self):

        if self._has_global_stats:
            if self.global_num_inst == 0:
                return self.name, float("nan")
            else:
                return self.name, self.global_sum_metric / self.global_num_inst
        else:
            return self.get()

    def get_name_value(self):

        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))

    def get_global_name_value(self):

        if self._has_global_stats:
            name, value = self.get_global()
            if not isinstance(name, list):
                name = [name]
            if not isinstance(value, list):
                value = [value]
            return list(zip(name, value))
        else:
            return self.get_name_value()


class CompositeEvalMetric(EvalMetric):

    def __init__(self,
                 name="composite",
                 output_names=None,
                 label_names=None):
        super(CompositeEvalMetric, self).__init__(
            name,
            output_names=output_names,
            label_names=label_names,
            has_global_stats=True)
        self.metrics = []

    def add(self, metric):

        self.metrics.append(metric)

    def update_dict(self, labels, preds):
        if self.label_names is not None:
            labels = OrderedDict([i for i in labels.items()
                                  if i[0] in self.label_names])
        if self.output_names is not None:
            preds = OrderedDict([i for i in preds.items()
                                 if i[0] in self.output_names])

        for metric in self.metrics:
            metric.update_dict(labels, preds)

    def update(self, labels, preds):

        for metric in self.metrics:
            metric.update(labels, preds)

    def reset(self):

        try:
            for metric in self.metrics:
                metric.reset()
        except AttributeError:
            pass

    def reset_local(self):

        try:
            for metric in self.metrics:
                metric.reset_local()
        except AttributeError:
            pass

    def get(self):

        names = []
        values = []
        for metric in self.metrics:
            name, value = metric.get()
            name = [name]
            value = [value]
            names.extend(name)
            values.extend(value)
        return names, values

    def get_global(self):

        names = []
        values = []
        for metric in self.metrics:
            name, value = metric.get_global()
            name = [name]
            value = [value]
            names.extend(name)
            values.extend(value)
        return names, values

    def get_config(self):
        config = super(CompositeEvalMetric, self).get_config()
        config.update({"metrics": [i.get_config() for i in self.metrics]})
        return config


class Accuracy(EvalMetric):

    def __init__(self,
                 axis=1,
                 name="accuracy",
                 output_names=None,
                 label_names=None):
        super(Accuracy, self).__init__(
            name,
            axis=axis,
            output_names=output_names,
            label_names=label_names,
            has_global_stats=True)
        self.axis = axis

    def update(self, labels, preds):

        assert (len(labels) == len(preds))
        with torch.no_grad():
            if preds.shape != labels.shape:
                pred_label = torch.argmax(preds, dim=self.axis)
            else:
                pred_label = preds
            pred_label = pred_label.cpu().numpy().astype(np.int32)
            label = labels.cpu().numpy().astype(np.int32)

            label = label.flat
            pred_label = pred_label.flat

            num_correct = (pred_label == label).sum()
            self.sum_metric += num_correct
            self.global_sum_metric += num_correct
            self.num_inst += len(pred_label)
            self.global_num_inst += len(pred_label)


class TopKAccuracy(EvalMetric):

    def __init__(self,
                 top_k=1,
                 name="top_k_accuracy",
                 torch_like=True,
                 output_names=None,
                 label_names=None):
        super(TopKAccuracy, self).__init__(
            name,
            top_k=top_k,
            output_names=output_names,
            label_names=label_names,
            has_global_stats=True)
        self.top_k = top_k
        assert (self.top_k > 1), "Please use Accuracy if top_k is no more than 1"
        self.name += "_{:d}".format(self.top_k)
        self.torch_like = torch_like

    def update(self, labels, preds):

        assert (len(labels) == len(preds))
        with torch.no_grad():
            if self.torch_like:
                _, pred = preds.topk(k=self.top_k, dim=1, largest=True, sorted=True)
                pred = pred.t()
                correct = pred.eq(labels.view(1, -1).expand_as(pred))
                num_correct = correct.view(-1).float().sum(dim=0, keepdim=True).item()
                num_samples = labels.size(0)
                assert (num_correct <= num_samples)
                self.sum_metric += num_correct
                self.global_sum_metric += num_correct
                self.num_inst += num_samples
                self.global_num_inst += num_samples
            else:
                assert(len(preds.shape) <= 2), "Predictions should be no more than 2 dims"
                pred_label = preds.cpu().numpy().astype(np.int32)
                pred_label = np.argpartition(pred_label, -self.top_k)
                label = labels.cpu().numpy().astype(np.int32)
                assert (len(label) == len(pred_label))
                num_samples = pred_label.shape[0]
                num_dims = len(pred_label.shape)
                if num_dims == 1:
                    num_correct = (pred_label.flat == label.flat).sum()
                    self.sum_metric += num_correct
                    self.global_sum_metric += num_correct
                elif num_dims == 2:
                    num_classes = pred_label.shape[1]
                    top_k = min(num_classes, self.top_k)
                    for j in range(top_k):
                        num_correct = (pred_label[:, num_classes - 1 - j].flat == label.flat).sum()
                        self.sum_metric += num_correct
                        self.global_sum_metric += num_correct
                self.num_inst += num_samples
                self.global_num_inst += num_samples


class Top1Error(Accuracy):

    def __init__(self,
                 axis=1,
                 name="top_1_error",
                 output_names=None,
                 label_names=None):
        super(Top1Error, self).__init__(
            axis=axis,
            name=name,
            output_names=output_names,
            label_names=label_names)

    def get(self):

        if self.num_inst == 0:
            return self.name, float("nan")
        else:
            return self.name, 1.0 - self.sum_metric / self.num_inst


class TopKError(TopKAccuracy):

    def __init__(self,
                 top_k=1,
                 name="top_k_error",
                 torch_like=True,
                 output_names=None,
                 label_names=None):
        name_ = name
        super(TopKError, self).__init__(
            top_k=top_k,
            name=name,
            torch_like=torch_like,
            output_names=output_names,
            label_names=label_names)
        self.name = name_.replace("_k_", "_{}_".format(top_k))

    def get(self):

        if self.num_inst == 0:
            return self.name, float("nan")
        else:
            return self.name, 1.0 - self.sum_metric / self.num_inst


class PixelAccuracyMetric(EvalMetric):

    def __init__(self,
                 axis=1,
                 name="pix_acc",
                 output_names=None,
                 label_names=None,
                 on_cpu=True,
                 sparse_label=True,
                 vague_idx=-1,
                 use_vague=False,
                 macro_average=True):
        self.macro_average = macro_average
        super(PixelAccuracyMetric, self).__init__(
            name,
            axis=axis,
            output_names=output_names,
            label_names=label_names)
        self.axis = axis
        self.on_cpu = on_cpu
        self.sparse_label = sparse_label
        self.vague_idx = vague_idx
        self.use_vague = use_vague

    def update(self, labels, preds):

        with torch.no_grad():
            check_label_shapes(labels, preds)
            if self.on_cpu:
                if self.sparse_label:
                    label_imask = labels.cpu().numpy().astype(np.int32)
                else:
                    label_imask = torch.argmax(labels, dim=self.axis).cpu().numpy().astype(np.int32)
                pred_imask = torch.argmax(preds, dim=self.axis).cpu().numpy().astype(np.int32)
                acc = seg_pixel_accuracy_np(
                    label_imask=label_imask,
                    pred_imask=pred_imask,
                    vague_idx=self.vague_idx,
                    use_vague=self.use_vague,
                    macro_average=self.macro_average)
                if self.macro_average:
                    self.sum_metric += acc
                    self.num_inst += 1
                else:
                    self.sum_metric += acc[0]
                    self.num_inst += acc[1]
            else:
                assert False

    def reset(self):

        if self.macro_average:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = 0
            self.sum_metric = 0

    def get(self):

        if self.macro_average:
            if self.num_inst == 0:
                return self.name, float("nan")
            else:
                return self.name, self.sum_metric / self.num_inst
        else:
            if self.num_inst == 0:
                return self.name, float("nan")
            else:
                return self.name, float(self.sum_metric) / self.num_inst


class MeanIoUMetric(EvalMetric):

    def __init__(self,
                 axis=1,
                 name="mean_iou",
                 output_names=None,
                 label_names=None,
                 on_cpu=True,
                 sparse_label=True,
                 num_classes=None,
                 vague_idx=-1,
                 use_vague=False,
                 bg_idx=-1,
                 ignore_bg=False,
                 macro_average=True):
        self.macro_average = macro_average
        self.num_classes = num_classes
        self.ignore_bg = ignore_bg
        super(MeanIoUMetric, self).__init__(
            name,
            axis=axis,
            output_names=output_names,
            label_names=label_names)
        assert ((not ignore_bg) or (bg_idx in (0, num_classes - 1)))
        self.axis = axis
        self.on_cpu = on_cpu
        self.sparse_label = sparse_label
        self.vague_idx = vague_idx
        self.use_vague = use_vague
        self.bg_idx = bg_idx

        assert (on_cpu and sparse_label)

    def update(self, labels, preds):

        assert (len(labels) == len(preds))
        with torch.no_grad():
            if self.on_cpu:
                if self.sparse_label:
                    label_imask = labels.cpu().numpy().astype(np.int32)
                else:
                    assert False
                pred_imask = torch.argmax(preds, dim=self.axis).cpu().numpy().astype(np.int32)
                batch_size = labels.shape[0]
                for k in range(batch_size):
                    if self.sparse_label:
                        acc = seg_mean_iou_imasks_np(
                            label_imask=label_imask[k, :, :],
                            pred_imask=pred_imask[k, :, :],
                            num_classes=self.num_classes,
                            vague_idx=self.vague_idx,
                            use_vague=self.use_vague,
                            bg_idx=self.bg_idx,
                            ignore_bg=self.ignore_bg,
                            macro_average=self.macro_average)
                    else:
                        assert False
                    if self.macro_average:
                        self.sum_metric += acc
                        self.num_inst += 1
                    else:
                        self.area_inter += acc[0]
                        self.area_union += acc[1]
            else:
                assert False

    def reset(self):

        if self.macro_average:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            class_count = self.num_classes - 1 if self.ignore_bg else self.num_classes
            self.area_inter = np.zeros((class_count,), np.uint64)
            self.area_union = np.zeros((class_count,), np.uint64)

    def get(self):

        if self.macro_average:
            if self.num_inst == 0:
                return self.name, float("nan")
            else:
                return self.name, self.sum_metric / self.num_inst
        else:
            class_count = (self.area_union > 0).sum()
            if class_count == 0:
                return self.name, float("nan")
            eps = np.finfo(np.float32).eps
            area_union_eps = self.area_union + eps
            mean_iou = (self.area_inter / area_union_eps).sum() / class_count
            return self.name, mean_iou


def seg_pixel_accuracy_np(label_imask,
                          pred_imask,
                          vague_idx=-1,
                          use_vague=False,
                          macro_average=True,
                          empty_result=0.0):

    assert (label_imask.shape == pred_imask.shape)
    if use_vague:
        sum_u_ij = np.sum(label_imask.flat != vague_idx)
        if sum_u_ij == 0:
            if macro_average:
                return empty_result
            else:
                return 0, 0
        sum_u_ii = np.sum(np.logical_and(pred_imask.flat == label_imask.flat, label_imask.flat != vague_idx))
    else:
        sum_u_ii = np.sum(pred_imask.flat == label_imask.flat)
        sum_u_ij = pred_imask.size
    if macro_average:
        return float(sum_u_ii) / sum_u_ij
    else:
        return sum_u_ii, sum_u_ij


def seg_mean_iou_imasks_np(label_imask,
                           pred_imask,
                           num_classes,
                           vague_idx=-1,
                           use_vague=False,
                           bg_idx=-1,
                           ignore_bg=False,
                           macro_average=True,
                           empty_result=0.0):
    assert (len(label_imask.shape) == 2)
    assert (len(pred_imask.shape) == 2)
    assert (pred_imask.shape == label_imask.shape)

    min_i = 1
    max_i = num_classes
    n_bins = num_classes

    if ignore_bg:
        n_bins -= 1
        if bg_idx != 0:
            assert (bg_idx == num_classes - 1)
            max_i -= 1

    if not (ignore_bg and (bg_idx == 0)):
        label_imask += 1
        pred_imask += 1
        vague_idx += 1

    if use_vague:
        label_imask = label_imask * (label_imask != vague_idx)
        pred_imask = pred_imask * (pred_imask != vague_idx)

    intersection = pred_imask * (pred_imask == label_imask)

    area_inter, _ = np.histogram(intersection, bins=n_bins, range=(min_i, max_i))
    area_pred, _ = np.histogram(pred_imask, bins=n_bins, range=(min_i, max_i))
    area_label, _ = np.histogram(label_imask, bins=n_bins, range=(min_i, max_i))
    area_union = area_pred + area_label - area_inter

    assert ((not ignore_bg) or (len(area_inter) == num_classes - 1))
    assert (ignore_bg or (len(area_inter) == num_classes))

    if macro_average:
        class_count = (area_union > 0).sum()
        if class_count == 0:
            return empty_result
        eps = np.finfo(np.float32).eps
        area_union = area_union + eps
        mean_iou = (area_inter / area_union).sum() / class_count
        return mean_iou
    else:
        return area_inter.astype(np.uint64), area_union.astype(np.uint64)
