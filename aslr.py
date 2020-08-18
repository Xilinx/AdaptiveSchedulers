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

import torch
from torch.optim.optimizer import Optimizer


class ASLR(object):
    """Adjusting learning rate based on validation loss,
    a greedy search adjusts the center of a uniform distribution
    for learning rates per layer.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_lr (float): An upper bound on the learning rate of
            all param groups. Default: None.
        min_lr (float): A lower bound on the learning rate of
            all param groups. Default: None.

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ASLR(optimizer)
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.update_loss(val_loss)  # this is after each epoch
        >>>     scheduler.step()  # we suggest to step the scheduler per mini-batch inside "train()"
    """

    def __init__(self, optimizer, max_lr=None, min_lr=None):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        self.lr_max = max_lr
        self.lr_min = min_lr

        self.min_loss = None

        self.lr_int, self.lr_scale = self.get_scale(self.optimizer.param_groups[0]['lr'])

        self.direction = 1  # -1: decay, 1 increase
        self.radius = None
        self.walk_counter = 0

    def get_scale(self, val):  # returns int and floating scale for val
        p = 0
        while True:
            if int(val) >= 1:
                return int(val), 1 / (10 ** p)
            else:
                val = val * 10
                p += 1

    def walk(self):
        if self.direction == -1:  # decreasing
            if self.lr_int == 1:
                self.lr_int = 9
                self.lr_scale /= 10
            else:
                self.lr_int -= 1
            if self.lr_min is not None:
                if self.lr_int * self.lr_scale < self.lr_min:
                    self.lr_int, self.lr_scale = self.get_scale(self.lr_min)

        elif self.direction == 1:  # increase
            if self.lr_int == 9:
                self.lr_int = 1
                self.lr_scale *= 10
            else:
                self.lr_int += 1
            if self.lr_max is not None:
                if self.lr_int * self.lr_scale > self.lr_max:
                    self.lr_int, self.lr_scale = self.get_scale(self.lr_max)

        else:
            raise print('direction: {} is not recognized'.format(self.direction))

    def adjust_direction(self):
        if self.radius is None:  # is it the first time?
            self.radius = 1  # start with one step radius
            self.walk_counter = 0  # reset walk counter
            self.direction = 1  # increase
        else:  # it is not the first time
            if self.walk_counter >= self.radius:  # if we reach the limit, increase the limit and change the direction
                self.radius += 1  # increase the limit
                self.direction *= -1  # change the direction
                self.walk_counter = 0  # reset walk counter
            else:  # we did not reach the limit
                pass
                # no change in radius
                # no change in direction

    def get_random_lr(self):
        _low = (self.lr_int - 0.5) * self.lr_scale
        _high = (self.lr_int + 0.5) * self.lr_scale
        return torch.FloatTensor(1).uniform_(_low, _high).item()

    def update_loss(self, val_loss):  # val_loss is validation loss
        if self.min_loss is None:  # is this the first time?
            self.min_loss = val_loss
        else:
            if val_loss > self.min_loss:  # our cost has been increased, we have to change the lr
                self.adjust_direction()
                self.walk()
                self.walk_counter += 1

            elif val_loss < self.min_loss:  # we got some improvement, keep up the current lr
                self.min_loss = val_loss
                self.radius = None  # reset direction settings

            else:  # we assume it never happens and we do not do anything for now
                print('No change in validation loss! you are either unlucky or very unlucky! check the accuracy now!')

    def step(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_random_lr()

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
