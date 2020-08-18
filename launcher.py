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

work_dir = 'data'
tensor_dir = 'experiments/tb_logs/'  # tensor board data

if not os.path.exists(tensor_dir):
    os.makedirs(tensor_dir)
    
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

for lr_mode in ['aslr', 'multistep']:
      cmd = 'PYTORCH_JIT=1 python trainer.py ' \
            '--num-gpus 1 ' \
            '--tensor-dir-path ' + tensor_dir + lr_mode + ' ' \
            '--work-dir ' + work_dir + ' ' \
            '--batch-size 128 ' \
            '--seed 123 ' \
            '-j 12 ' \
            '--num-epochs 100 ' \
            '--lr 0.1 ' \
            '--lr-mode ' + lr_mode + ' ' \
            '--lr-decay-epoch 80,120,160,180 ' \
            '--wd 0.00001 ' \
            '--momentum 0.9 ' \
            '--optimizer-name nag'

      tic = time.time()
      os.system(cmd)
      toc = time.time()
      print('finished in {} minutes.'.format((toc-tic)/60))
