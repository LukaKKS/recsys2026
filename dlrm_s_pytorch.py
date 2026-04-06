# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Description: an implementation of a deep learning recommendation model (DLRM)
# The model input consists of dense and sparse features. The former is a vector
# of floating point values. The latter is a list of sparse indices into
# embedding tables, which consist of vectors of floating point values.
# The selected vectors are passed to mlp networks denoted by triangles,
# in some cases the vectors are interacted through operators (Ops).
#
# output:
#                         vector of values
# model:                        |
#                              /\
#                             /__\
#                               |
#       _____________________> Op  <___________________
#     /                         |                      \
#    /\                        /\                      /\
#   /__\                      /__\           ...      /__\
#    |                          |                       |
#    |                         Op                      Op
#    |                    ____/__\_____           ____/__\____
#    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
# input:
# [ dense features ]     [sparse indices] , ..., [sparse indices]
#
# More precise definition of model layers:
# 1) fully connected layers of an mlp
# z = f(y)
# y = Wx + b
#
# 2) embedding lookup (for a list of sparse indices p=[p1,...,pk])
# z = Op(e1,...,ek)
# obtain vectors e1=E[:,p1], ..., ek=E[:,pk]
#
# 3) Operator Op can be one of the following
# Sum(e1,...,ek) = e1 + ... + ek
# Dot(e1,...,ek) = [e1'e1, ..., e1'ek, ..., ek'e1, ..., ek'ek]
# Cat(e1,...,ek) = [e1', ..., ek']'
# where ' denotes transpose operation
#
# References:
# [1] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
# Narayanan Sundaram, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu,
# Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii,
# Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko,
# Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong,
# Misha Smelyanskiy, "Deep Learning Recommendation Model for Personalization and
# Recommendation Systems", CoRR, arXiv:1906.00091, 2019

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

# miscellaneous
import builtins
import datetime
import json
import sys
import time
import ctypes
import math
import profile
import os
import concurrent.futures

# onnx
# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings

# data generation
import dlrm_data_pytorch as dp
import huffman_coding as hc
import adaptive_encoding as ae

# CLS Cold Start 모듈
from transfer_module import TransferModule
from dynamic_lr import DynamicLR

# numpy
import numpy as np
import sklearn.metrics

# pytorch
import torch
import torch.nn as nn
from torch._ops import ops
from torch.autograd.profiler import record_function
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
import random
import pickle
import multiprocessing

# quotient-remainder trick
from tricks.qr_embedding_bag import QREmbeddingBag
from tricks.sk_embedding_bag import SKEmbeddingBag
from tricks.md_embedding_bag import PrEmbeddingBag
from tricks.md_embedding_bag import md_solver
from tricks.sk_embedding_bag import get_sketch_time
from tricks.sk_embedding_bag import reset_sketch_time
from tricks.adaembed import adaEmbeddingBag

from collections import defaultdict
from datetime import datetime
from typing import List
from joblib import Parallel, delayed
from functools import partial
from itertools import combinations
from hash_embedding import HashEmbedding


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        import onnx
    except ImportError as error:
        print("Unable to import onnx. ", error)

# from torchviz import make_dot
# import torch.nn.functional as Functional
# from torch.nn.parameter import Parameter

exc = getattr(builtins, "IOError", "FileNotFoundError")


def time_wrap(use_gpu):
    if use_gpu:
        torch.cuda.synchronize()
    return time.time()


def dlrm_wrap(
    X,
    lS_o,
    lS_i,
    use_gpu,
    device,
    ndevices,
    test,
    sk_flag,
    use_huffman_coding,
    huffman_coding_tensors,
    max_lens,
    ln_emb,
    capacity,
    use_hashing_trick,
    is_reversed,
    reverse_and_shift_huffman_coding,
    num_huffman_digits,
    use_adaptive_encoding,
    online_frequency_checker,
    compressed_table_mask,
    selected_ln_emb_cum_offsets,
    use_hash_embedding,
    hash_embeddings,
    num_buckets,
    short_head_hash,
    long_tail_hash,
    iteration_index,
    frequency_percentile,
    short_head_indices_set=set(),
    train_time=None,
    transfer_module=None,
    surge_long_tail_hash=None,
):
    with record_function("DLRM forward"):
        # --- CPU-mode adaptive encoding (runs when use_gpu=False, e.g. Apple Silicon) ---
        if use_adaptive_encoding and not use_gpu:
            # lS_i may be a stacked tensor or a list; handle both.
            if isinstance(lS_i, torch.Tensor):
                selected = lS_i[compressed_table_mask]
            else:
                mask_indices = [k for k, m in enumerate(compressed_table_mask) if m]
                selected = torch.stack([lS_i[k] for k in mask_indices])
            encoded_indices, encoded_offsets = ae.batch_adaptive_encoding_with_hashing(
                selected,
                online_frequency_checker,
                selected_ln_emb_cum_offsets,
                device,
                long_tail_hash,
                short_head_hash,
                iteration_index,
                frequency_percentile,
                short_head_indices_set,
                transfer_module=transfer_module,
                surge_long_tail_hash=surge_long_tail_hash,
            )
            lS_i_tmp = []
            j = 0
            for k, sparse_index_group_batch in enumerate(lS_i):
                if compressed_table_mask[k]:
                    lS_i_tmp.append(encoded_indices[j])
                    j += 1
                else:
                    lS_i_tmp.append(sparse_index_group_batch)
            lS_o[compressed_table_mask] = encoded_offsets
            lS_i = lS_i_tmp
        # --------------------------------------------------------------------------

        if use_gpu:
            # lS_i can be either a list of tensors or a stacked tensor.
            # Handle each case below:
            if ndevices == 1:
                if sk_flag:
                    if X != None:
                        return dlrm(X.to(device), lS_o, lS_i, test, train_time)
                    else:
                        return dlrm(None, lS_o, lS_i, test, train_time)
                else:
                    if use_huffman_coding or use_hashing_trick:
                        if capacity < 50:
                            print("capacity too small")

                    if use_adaptive_encoding:
                        selected_sparse_index_group_batch_tensor = lS_i[compressed_table_mask].to(device)
                        encoded_indices, encoded_offsets = ae.batch_adaptive_encoding_with_hashing(
                            selected_sparse_index_group_batch_tensor,
                            online_frequency_checker,
                            selected_ln_emb_cum_offsets,
                            device,
                            long_tail_hash,
                            short_head_hash,
                            iteration_index,
                            frequency_percentile,
                            short_head_indices_set,
                            transfer_module=transfer_module,
                            surge_long_tail_hash=surge_long_tail_hash,
                        )

                        lS_i_tmp = []

                        j = 0
                        for k, sparse_index_group_batch in enumerate(lS_i):
                            if compressed_table_mask[k]:
                                lS_i_tmp.append(encoded_indices[j])
                                j += 1
                            else:
                                lS_i_tmp.append(sparse_index_group_batch)

                        lS_o[compressed_table_mask] = encoded_offsets.cpu()

                        lS_i = lS_i_tmp

                    else:
                        lS_i_tmp = []
                        lS_o_tmp = []

                        for k, sparse_index_group_batch in enumerate(lS_i):
                            emb_index_group_batch = sparse_index_group_batch
                            emb_offset_group_batch = lS_o[k]
                            if use_huffman_coding and ln_emb[k] > capacity:
                                table_size = max_lens[k] * num_huffman_digits + 1
                                emb_index_group_batch, emb_offset_group_batch = hc.huffman_coding_lookup(sparse_index_group_batch, huffman_coding_tensors[k], is_reversed, reverse_and_shift_huffman_coding, table_size)
                            elif use_hashing_trick:
                                emb_index_group_batch %= capacity
                            elif use_hash_embedding and ln_emb[k] > capacity:
                                if hash_embeddings[k] is None:
                                    print(f"ERROR: hash_embeddings[{k}] cannot be None")
                                emb_index_group_batch, emb_offset_group_batch = hash_embeddings[k].get_hash_embedding_indices(sparse_index_group_batch)

                            lS_i_tmp.append(emb_index_group_batch)
                            lS_o_tmp.append(emb_offset_group_batch)

                        lS_i = lS_i_tmp
                        lS_o = lS_o_tmp

                    lS_i = (
                        [S_i.to(device) for S_i in lS_i]
                        if isinstance(lS_i, list)
                        else lS_i.to(device)
                    )
                    lS_o = (
                        [S_o.to(device) for S_o in lS_o]
                        if isinstance(lS_o, list)
                        else lS_o.to(device)
                    )
        if X != None:
            return dlrm(X.to(device), lS_o, lS_i, test, train_time)
        else:
            return dlrm(None, lS_o, lS_i, test, train_time)


def loss_fn_wrap(Z, T, use_gpu, device):
    with record_function("DLRM loss compute"):
        if args.loss_function == "mse" or args.loss_function == "bce":
            return dlrm.loss_fn(Z, T.to(device))
        elif args.loss_function == "wbce":
            loss_ws_ = dlrm.loss_ws[T.data.view(-1).long()
                                    ].view_as(T).to(device)
            loss_fn_ = dlrm.loss_fn(Z, T.to(device))
            loss_sc_ = loss_ws_ * loss_fn_
            return loss_sc_.mean()


# The following function is a wrapper to avoid checking this multiple times in th
# loop below.
def unpack_batch(b):
    # Experiment with unweighted samples
    return b[0], b[1], b[2], b[3], torch.ones(b[3].size()), None


class LRPolicyScheduler(_LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, decay_start_step, num_decay_steps):
        self.num_warmup_steps = num_warmup_steps
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_start_step + num_decay_steps
        self.num_decay_steps = num_decay_steps

        if self.decay_start_step < self.num_warmup_steps:
            sys.exit("Learning rate warmup must finish before the decay starts")

        super(LRPolicyScheduler, self).__init__(optimizer)

    def get_lr(self):
        step_count = self._step_count
        if step_count < self.num_warmup_steps:
            # warmup
            scale = 1.0 - (self.num_warmup_steps - step_count) / \
                self.num_warmup_steps
            lr = [base_lr * scale for base_lr in self.base_lrs]
            self.last_lr = lr
        elif self.decay_start_step <= step_count and step_count < self.decay_end_step:
            # decay
            decayed_steps = step_count - self.decay_start_step
            scale = ((self.num_decay_steps - decayed_steps) /
                     self.num_decay_steps) ** 2
            min_lr = 0.0000001
            lr = [max(min_lr, base_lr * scale) for base_lr in self.base_lrs]
            self.last_lr = lr
        else:
            if self.num_decay_steps > 0:
                # freeze at last, either because we're after decay
                # or because we're between warmup and decay
                lr = self.last_lr
            else:
                # do not adjust
                lr = self.base_lrs
        return lr


### define dlrm in PyTorch ###
class DLRM_Net(nn.Module):
    def create_mlp(self, ln, sigmoid_layer):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers
        return torch.nn.Sequential(*layers)

    # def create_emb(self, m, ln, weighted_pooling=None, max_lens=None, use_huffman_coding=False, use_hashing_trick=False, num_huffman_emb_tables):
    def create_emb(
        self,
        m: int,
        ln: List[int],
        weighted_pooling: str,
        max_lens: List[int],
        use_huffman_coding: bool,
        use_hashing_trick: bool,
        num_huffman_digits: int,
        use_adaptive_encoding: bool,
        use_hash_embedding: bool,
        num_buckets: int,
    ):
        emb_l = nn.ModuleList()
        print(ln)
        v_W_l = []
        tot_nums = 0
        N = 0
        if self.ada_flag:
            for i in range(0, ln.size):
                if ln[i] > 2000 * self.compress_rate:
                    N += ln[i]
            self.dic = np.zeros(N, dtype=np.int32)
            self.hotn = int((N * m * self.compress_rate - N * 2) / m)
            print(f"N: {N}, m: {m}, compress_rate: {self.compress_rate}, hotn: {self.hotn}")
            self.weight = Parameter(
                torch.Tensor(self.hotn + 1, m),
                requires_grad=True,
            )
            self.hot_rate = self.hotn / N
            self.grad_norm = np.zeros(N, dtype=np.float64)
            self.tmp_grad_norm = np.zeros(N, dtype=np.float64)
            print(f"hotn: {self.hotn}, hot_rate: {self.hot_rate}")
        N = 0
        print(f"ln: {ln}")

        total_num_trainable_parameters = 0

        for i in range(0, ln.size):
            n = ln[i]

            if use_huffman_coding and ln[i] > self.capacity:
                n = max_lens[i] * num_huffman_digits + 1
            elif use_hashing_trick:
                n = min(ln[i], self.capacity)
            elif use_adaptive_encoding and ln[i] > self.capacity:
                n = self.capacity + 5
            elif use_hash_embedding and ln[i] > self.capacity:
                n = self.capacity

            print(f"embedding table {i} number of trainable parameters: {n * m}")
            total_num_trainable_parameters += n * m

            # construct embedding operator
            if self.qr_flag and n > self.qr_threshold:
                EE = QREmbeddingBag(
                    n,
                    m,
                    self.qr_collisions,
                    operation=self.qr_operation,
                    mode="sum",
                    sparse=True,
                )
            elif self.md_flag and n > self.md_threshold:
                base = max(m)
                _m = m[i] if n > self.md_threshold else base
                EE = PrEmbeddingBag(n, _m, base)
                # use np initialization as below for consistency...
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, _m)
                ).astype(np.float32)
                EE.embs.weight.data = torch.tensor(W, requires_grad=True)
            elif self.ada_flag and n > 2000 * self.compress_rate:
                EE = adaEmbeddingBag(
                    N,
                    self.weight,
                    self.dic,
                    self.compress_rate,
                    self.device,
                    n,
                    m,
                )
                self.ada_emb[i] = True
                self.f_offset[i] = N
                N += n
            elif self.sketch_flag and n > 2000 * self.compress_rate:
                EE = SKEmbeddingBag(
                    N,
                    self.hotn,
                    self.lib,
                    self.weight_high,
                    self.device,
                    n,
                    m,
                    tot_nums,
                    round(self.hash_rate * n + 0.51),
                )
                self.sketch_emb[i] = True
                N += 1
            else:
                if self.md_flag:
                    EE = nn.EmbeddingBag(n, max(m), mode="sum", sparse=True)
                else:
                    EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
                # initialize embeddings
                # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
                tmp_n = max(n, 5)
                
                if self.md_flag:
                    W = np.random.uniform(
                        low=-np.sqrt(1 / tmp_n), high=np.sqrt(1 / tmp_n), size=(n, max(m))
                    ).astype(np.float32)
                else:
                    W = np.random.uniform(
                        low=-np.sqrt(1 / tmp_n), high=np.sqrt(1 / tmp_n), size=(n, m)
                    ).astype(np.float32)
                # approach 1
                EE.weight.data = torch.tensor(W, requires_grad=True)
                # approach 2
                # EE.weight.data.copy_(torch.tensor(W))
                # approach 3
                # EE.weight = Parameter(torch.tensor(W),requires_grad=True)
            if weighted_pooling is None:
                v_W_l.append(None)
            else:
                v_W_l.append(torch.ones(n, dtype=torch.float32))
            emb_l.append(EE)
            if self.ada_flag == False:
                self.f_offset[i] = tot_nums
            tot_nums += n
        
        print(f"total_num_trainable_parameters: {total_num_trainable_parameters}")
        
        return emb_l, v_W_l

    def __init__(
        self,
        compress_rate,
        ada_flag,
        lib,
        hotn,
        device,
        sketch_flag,
        hash_rate,
        m_spa=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        arch_interaction_op=None,
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        sync_dense_params=True,
        loss_threshold=0.0,
        ndevices=-1,
        qr_flag=False,
        qr_operation="mult",
        qr_collisions=0,
        qr_threshold=200,
        md_flag=False,
        md_threshold=200,
        weighted_pooling=None,
        loss_function="bce",
        max_lens=None,
        use_huffman_coding=False,
        capacity=0,
        use_hashing_trick=False,
        num_huffman_digits=2,
        use_adaptive_encoding=False,
        use_hash_embedding=False,
        num_buckets=40
    ):
        super(DLRM_Net, self).__init__()

        if (
            (m_spa is not None)
            and (ln_emb is not None)
            and (ln_bot is not None)
            and (ln_top is not None)
            and (arch_interaction_op is not None)
        ):

            # save arguments
            self.ln_emb = ln_emb
            self.compress_rate = compress_rate
            self.ada_flag = ada_flag
            if self.ada_flag:
                self.dic = []
                self.weight = None
                self.hot_rate = 0
                self.hotn = 0
                self.d_time = 0
            self.ada_emb = np.zeros(len(self.ln_emb))
            self.lib = lib
            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            self.loss_function = loss_function
            self.device = device
            self.g_time = 0
            self.max_lens = max_lens
            self.capacity = capacity

            if weighted_pooling is not None and weighted_pooling != "fixed":
                self.weighted_pooling = "learned"
            else:
                self.weighted_pooling = weighted_pooling
            # create variables for QR embedding if applicable
            self.qr_flag = qr_flag
            if self.qr_flag:
                self.qr_collisions = qr_collisions
                self.qr_operation = qr_operation
                self.qr_threshold = qr_threshold
            # create variables for MD embedding if applicable
            self.md_flag = md_flag
            if self.md_flag:
                self.md_threshold = md_threshold
            self.sketch_flag = sketch_flag
            self.hash_rate = hash_rate
            self.hotn = hotn
            self.grad_norm = []
            self.f_offset = np.zeros(self.ln_emb.size, dtype=np.int32)
            if self.sketch_flag:
                self.weight_high = Parameter(
                    torch.Tensor(hotn, m_spa),
                    requires_grad=True,
                )
                self.register_buffer('sketch_buffer', torch.zeros(hotn * 12, dtype = torch.int32, device = 'cpu'))
                init = lib.init
                init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
                init.restype = None
                numpy_array = self.sketch_buffer.numpy()
                data_ptr = numpy_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
                init(hotn, args.sketch_threshold, data_ptr)

            self.sketch_emb = np.zeros(self.ln_emb.size)
            # nn.init.uniform_(self.weight_high, np.sqrt(1 / 10000000))
            # If running distributed, get local slice of embedding tables

            # create operators
            if ndevices <= 1:
                self.emb_l, w_list = self.create_emb(
                    m_spa,
                    ln_emb,
                    weighted_pooling,
                    max_lens,
                    use_huffman_coding,
                    use_hashing_trick,
                    num_huffman_digits,
                    use_adaptive_encoding,
                    use_hash_embedding,
                    num_buckets
                )
                if self.weighted_pooling == "learned":
                    self.v_W_l = nn.ParameterList()
                    for w in w_list:
                        self.v_W_l.append(Parameter(w))
                else:
                    self.v_W_l = w_list
            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
            self.top_l = self.create_mlp(ln_top, sigmoid_top)

            # quantization
            self.quantize_emb = False
            self.emb_l_q = []
            self.quantize_bits = 32

            # specify the loss function
            if self.loss_function == "mse":
                self.loss_fn = torch.nn.MSELoss(reduction="mean")
            elif self.loss_function == "bce":
                self.loss_fn = torch.nn.BCELoss(reduction="mean")
            elif self.loss_function == "wbce":
                self.loss_ws = torch.tensor(
                    np.fromstring(args.loss_weights, dtype=float, sep="-")
                )
                self.loss_fn = torch.nn.BCELoss(reduction="none")
            else:
                sys.exit(
                    "ERROR: --loss-function=" + self.loss_function + " is not supported"
                )
    
    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        def compute_should_use_set_data(tensor, tensor_applied):
            if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
                # If the new tensor has compatible tensor type as the existing tensor,
                # the current behavior is to change the tensor in-place using `.data =`,
                # and the future behavior is to overwrite the existing tensor. However,
                # changing the current behavior is a BC-breaking change, and we want it
                # to happen in future releases. So for now we introduce the
                # `torch.__future__.get_overwrite_module_params_on_conversion()`
                # global flag to let the user control whether they want the future
                # behavior of overwriting the existing tensor or not.
                return not torch.__future__.get_overwrite_module_params_on_conversion()
            else:
                return False

        for key, param in self._parameters.items():
            if param is None:
                continue
            # Tensors stored in modules are graph leaves, and we don't want to
            # track autograd history of `param_applied`, so we have to use
            # `with torch.no_grad():`
            with torch.no_grad():
                param_applied = fn(param)
            should_use_set_data = compute_should_use_set_data(param, param_applied)
            if should_use_set_data:
                param.data = param_applied
                out_param = param
            else:
                assert isinstance(param, Parameter)
                assert param.is_leaf
                out_param = Parameter(param_applied, param.requires_grad)
                self._parameters[key] = out_param

            if param.grad is not None:
                with torch.no_grad():
                    grad_applied = fn(param.grad)
                should_use_set_data = compute_should_use_set_data(param.grad, grad_applied)
                if should_use_set_data:
                    assert out_param.grad is not None
                    out_param.grad.data = grad_applied
                else:
                    assert param.grad.is_leaf
                    out_param.grad = grad_applied.requires_grad_(param.grad.requires_grad)

        for key, buf in self._buffers.items():
            if buf is not None and key != "sketch_buffer":
                self._buffers[key] = fn(buf)

        return self

    def apply_mlp(self, x, layers):
        # approach 1: use ModuleList
        # for layer in layers:
        #     x = layer(x)
        # return x
        # approach 2: use Sequential container to wrap all layers
        if x == None:
            return None
        return layers(x)

    def apply_emb(self, lS_o, lS_i, emb_l, v_W_l, test):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups

        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            # E = emb_l[k]

            if v_W_l[k] is not None:
                per_sample_weights = v_W_l[k].gather(
                    0, sparse_index_group_batch)
            else:
                per_sample_weights = None

            if self.quantize_emb:
                s1 = self.emb_l_q[k].element_size() * \
                    self.emb_l_q[k].nelement()
                s2 = self.emb_l_q[k].element_size() * \
                    self.emb_l_q[k].nelement()
                print("quantized emb sizes:", s1, s2)

                if self.quantize_bits == 4:
                    QV = ops.quantized.embedding_bag_4bit_rowwise_offsets(
                        self.emb_l_q[k],
                        sparse_index_group_batch,
                        sparse_offset_group_batch,
                        per_sample_weights=per_sample_weights,
                    )
                elif self.quantize_bits == 8:
                    QV = ops.quantized.embedding_bag_byte_rowwise_offsets(
                        self.emb_l_q[k],
                        sparse_index_group_batch,
                        sparse_offset_group_batch,
                        per_sample_weights=per_sample_weights,
                    )

                ly.append(QV)
            else:
                E = emb_l[k]
                if (self.sketch_emb[k] == True):
                    V = E(
                        sparse_index_group_batch,
                        sparse_offset_group_batch,
                        per_sample_weights=per_sample_weights,
                        test=test,
                    )
                else:

                    V = E(
                        sparse_index_group_batch.to(self.device),
                        sparse_offset_group_batch.to(self.device),
                        per_sample_weights=per_sample_weights,
                    )

                ly.append(V)

        # print(ly)
        return ly
    

    def ada_decay(self):
        self.grad_norm *= 0.8

    def ada_rebuild(self):
        new_cnt = self.grad_norm[:]
        for i in range(self.ln_emb.size):
            if self.ada_emb[i]:
                l = self.f_offset[i]
                r = self.f_offset[i] + self.ln_emb[i]
                p = np.percentile(self.grad_norm[l: r], 95)
                if (p != 0):
                    new_cnt[l: r] /= p
        ind_1 = set(np.argsort(-new_cnt)[:self.hotn])  # all hot features
        ind_2 = set(np.where(self.dic != 0)[0])  # old hot features
        admit_ = np.array(list(ind_1 - ind_2))
        evict_ = np.array(list(ind_2 - ind_1))
        if len(admit_) != len(evict_):
            if (len(evict_) != 0):
                print("error")
                exit(0)
            self.dic[admit_] = np.arange(1, self.hotn + 1)
        else:
            self.dic[admit_] = self.dic[evict_]
            with torch.no_grad():
                self.weight[self.dic[admit_]] = 0
            self.dic[evict_] = 0

    def ada_check(self):
        N = 1000000
        sample = random.sample(range(0, len(self.grad_norm)), N)
        dic = self.dic[sample]
        cnt = self.grad_norm[sample]
        m = math.ceil(N * self.hot_rate)

        ind_1 = set(np.argsort(-cnt)[:m])  # all hot features
        ind_2 = set(np.where(dic != 0)[0])  # old hot features
        admit_ = np.array(list(ind_1 - ind_2))
        evict_ = np.array(list(ind_2 - ind_1))
        lim = cnt[np.argsort(-cnt)[m]]  # all hot features
        if (len(admit_) > m * 0.05):
            print(f"evict: {len(admit_)}, m: {m}")
            self.ada_rebuild()

    def insert_adagrad(self, lS_o):
        tmp = 0
        N = len(lS_o[0])
        for k, input in enumerate(lS_o):
            if self.ada_emb[k] == True:
                grad_norm = np.array(torch.norm(self.weight.grad._values()[
                                     tmp: tmp+N], dim=1, p=2).cpu())
                np.add.at(self.grad_norm, input +
                          self.f_offset[k], grad_norm / np.sum(grad_norm) * N)
                # self.grad_norm[input+self.f_offset[k]] += grad_norm / np.sum(grad_norm) * N
                tmp += N
        self.d_time += 1
        if (self.d_time == 1 or self.d_time % 4096 == 0):
            self.ada_check()
        if (self.d_time % 16384 == 0):
            self.ada_decay()

    def insert_grad(self, lS_o):
        N = len(lS_o[0])
        for k, input in enumerate(lS_o):
            if self.sketch_emb[k] == True:
                self.emb_l[k].insert_grad(input)

    #  using quantizing functions from caffe2/aten/src/ATen/native/quantized/cpu
    def quantize_embedding(self, bits):

        n = len(self.emb_l)
        self.emb_l_q = [None] * n
        for k in range(n):
            if bits == 4:
                self.emb_l_q[k] = ops.quantized.embedding_bag_4bit_prepack(
                    self.emb_l[k].weight
                )
            elif bits == 8:
                self.emb_l_q[k] = ops.quantized.embedding_bag_byte_prepack(
                    self.emb_l[k].weight
                )
            else:
                return
        self.emb_l = None
        self.quantize_emb = True
        self.quantize_bits = bits

    def interact_features(self, x, ly):
        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            (batch_size, d) = ly[0].shape

            if x == None:
                T = torch.cat(ly, dim=1).view((batch_size, -1, d))
            else:
                T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            # append dense feature with the interactions (into a row vector)
            # approach 1: all
            # Zflat = Z.view((batch_size, -1))
            # approach 2: unique
            _, ni, nj = Z.shape
            # approach 1: tril_indices
            # offset = 0 if self.arch_interaction_itself else -1
            # li, lj = torch.tril_indices(ni, nj, offset=offset)
            # approach 2: custom
            offset = 1 if self.arch_interaction_itself else 0
            li = torch.tensor([i for i in range(ni)
                              for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj)
                              for j in range(i + offset)])
            # print(f"ni, nj: {ni, nj}, li, lj: {li, lj}")
            Zflat = Z[:, li, lj]
            # concatenate dense features and interactions
            if x != None:
                R = torch.cat([x] + [Zflat], dim=1)
            else:
                R = torch.cat([Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return R
    
    def forward(self, dense_x, lS_o, lS_i, test, train_time):
        return self.sequential_forward(dense_x, lS_o, lS_i, test, train_time)

    def sequential_forward(self, dense_x, lS_o, lS_i, test, train_time):
        # start the timer if it is valid
        if train_time is not None:
            train_time[0] = time.time()
        # process dense features (using bottom mlp), resulting in a row vector
        x = self.apply_mlp(dense_x, self.bot_l)
        # debug prints
        # print("intermediate")
        # print(x.detach().cpu().numpy())

        # process sparse features(using embeddings), resulting in a list of row vectors
        ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l, test)
        # for y in ly:
        #     print(y.detach().cpu().numpy())

        # interact features (dense and sparse)
        z = self.interact_features(x, ly)
        # print(z.detach().cpu().numpy())

        # obtain probability of a click (using top mlp)
        p = self.apply_mlp(z, self.top_l)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold,
                            max=(1.0 - self.loss_threshold))
        else:
            z = p

        return z


def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )

    return value


def dash_separated_floats(value):
    vals = value.split("-")
    for val in vals:
        try:
            float(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of floats" % value
            )

    return value


def inference(
    args,
    dlrm,
    best_metric,
    test_ld,
    device,
    use_gpu,
    log_iter,
    test,
    sk_flag,
    use_huffman_coding,
    huffman_coding_tensors,
    max_lens,
    ln_emb,
    capacity,
    use_hashing_trick,
    is_coding_reversed,
    reverse_and_shift_huffman_coding,
    num_huffman_digits,
    use_adaptive_encoding,
    online_frequency_checker,
    compressed_table_mask,
    selected_ln_emb_cum_offsets,
    use_hash_embedding,
    hash_embeddings,
    num_buckets,
    short_head_hash,
    long_tail_hash,
    iteration_index,
    frequency_percentile,
    short_head_indices_set
):
    test_accu = 0
    test_samp = 0
    scores = []
    targets = []
    
    for i, testBatch in enumerate(test_ld):
        # early exit if nbatches was set by the user and was exceeded
        if nbatches > 0 and i >= nbatches:
            break
        X_test, lS_o_test, lS_i_test, T_test, W_test, CBPP_test = unpack_batch(
            testBatch
        )
        # forward pass
        Z_test = dlrm_wrap(
            X_test,
            lS_o_test,
            lS_i_test,
            use_gpu,
            device,
            ndevices=ndevices,
            test=test,
            sk_flag=sk_flag,
            use_huffman_coding=use_huffman_coding,
            huffman_coding_tensors=huffman_coding_tensors,
            max_lens=max_lens,
            ln_emb=ln_emb,
            capacity=capacity,
            use_hashing_trick=use_hashing_trick,
            is_reversed=is_coding_reversed,
            reverse_and_shift_huffman_coding=reverse_and_shift_huffman_coding,
            num_huffman_digits=num_huffman_digits,
            use_adaptive_encoding=use_adaptive_encoding,
            online_frequency_checker=online_frequency_checker,
            compressed_table_mask=compressed_table_mask,
            selected_ln_emb_cum_offsets=selected_ln_emb_cum_offsets,
            use_hash_embedding=use_hash_embedding,
            hash_embeddings=hash_embeddings,
            num_buckets=num_buckets,
            short_head_hash=short_head_hash,
            long_tail_hash=long_tail_hash,
            iteration_index=iteration_index,
            frequency_percentile=frequency_percentile,
            short_head_indices_set=short_head_indices_set,
            transfer_module=None,
        )
        if Z_test.is_cuda:
            torch.cuda.synchronize()

        with record_function("DLRM accuracy compute"):
            # compute loss and accuracy

            S_test = Z_test.detach().cpu().numpy()  # numpy array
            T_test = T_test.detach().cpu().numpy()  # numpy array
            scores.append(S_test)
            targets.append(T_test)

            mbs_test = T_test.shape[0]  # = mini_batch_size except last
            A_test = np.sum(
                (np.round(S_test, 0) == T_test).astype(np.uint8))

            test_accu += A_test
            test_samp += mbs_test

    scores = np.concatenate(scores, axis=0)
    targets = np.concatenate(targets, axis=0)

    metrics = {
        "recall": lambda y_true, y_score: sklearn.metrics.recall_score(
            y_true=y_true, y_pred=np.round(y_score)
        ),
        "precision": lambda y_true, y_score: sklearn.metrics.precision_score(
            y_true=y_true, y_pred=np.round(y_score)
        ),
        "f1": lambda y_true, y_score: sklearn.metrics.f1_score(
            y_true=y_true, y_pred=np.round(y_score)
        ),
        "ap": sklearn.metrics.average_precision_score,
        "roc_auc": sklearn.metrics.roc_auc_score,
    }

    validation_results = {}
    for metric_name, metric_function in metrics.items():
        validation_results[metric_name] = metric_function(targets, scores)
        writer.add_scalar(
            metric_name,
            validation_results[metric_name],
            log_iter,
        )

    acc_test = test_accu / test_samp
    writer.add_scalar("Test/Acc", acc_test, log_iter)

    model_metrics_dict = {
        "nepochs": args.nepochs,
        "nbatches": nbatches,
        "nbatches_test": nbatches_test,
        "state_dict": dlrm.state_dict(),
        "test_acc": acc_test,
    }

    is_best_auc = validation_results["roc_auc"] > best_metric[1]
    if is_best_auc:
        best_metric[1] = validation_results["roc_auc"]
        model_metrics_dict["test_auc"] = best_metric[1]
    is_best_acc = acc_test > best_metric[0]
    if is_best_acc:
        best_metric[0] = acc_test

    print(
        " accuracy {:3.3f} %, best accuracy {:3.3f} %, auc {:3.3f} %, best auc {:3.3f} %".format(
            acc_test * 100,
            best_metric[0] * 100,
            validation_results['roc_auc'] * 100,
            best_metric[1] * 100,
        ),
        flush=True,
    )
    return model_metrics_dict, is_best_acc, is_best_auc


def get_capacity(compression_ratio: int, table_sizes: list) -> int:
    total_parameters = sum(table_sizes)
    budget = total_parameters // compression_ratio
    
    def find_capacity(budget: int):
        left, right = 1, 20_000_000
        while left <= right:
            mid = left + (right - left) // 2
            total = 0
            for sz in table_sizes:
                total += min(mid, sz)
            # total *= 16
            if total == budget:
                break
            elif total < budget:
                left = mid + 1
            else:
                right = mid - 1
        return left
    
    return find_capacity(budget)


def run():
    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    # model related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
    parser.add_argument(
        "--arch-embedding-size", type=dash_separated_ints, default="4-3-2"
    )
    # j will be replaced with the table number
    parser.add_argument(
        "--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
    parser.add_argument(
        "--arch-mlp-top", type=dash_separated_ints, default="4-2-1")
    parser.add_argument(
        "--arch-interaction-op", type=str, choices=["dot", "cat"], default="dot"
    )
    parser.add_argument("--arch-interaction-itself",
                        action="store_true", default=False)
    parser.add_argument("--weighted-pooling", type=str, default=None)
    # embedding table options
    parser.add_argument("--md-flag", action="store_true", default=False)
    parser.add_argument("--md-threshold", type=int, default=200)
    parser.add_argument("--md-temperature", type=float, default=0.3)
    parser.add_argument("--md-round-dims", action="store_true", default=False)
    parser.add_argument("--qr-flag", action="store_true", default=False)
    parser.add_argument("--qr-threshold", type=int, default=200)
    parser.add_argument("--qr-operation", type=str, default="mult")
    parser.add_argument("--qr-collisions", type=int, default=10)
    parser.add_argument("--lr-flag", action="store_true", default=False)
    parser.add_argument("--hash-flag", action="store_true", default=False)
    parser.add_argument("--bucket-flag", action="store_true", default=False)
    parser.add_argument("--sketch-flag", action="store_true", default=False)
    parser.add_argument("--compress-rate", type=float, default=0.001)
    parser.add_argument("--hc-threshold", type=int, default=200)
    parser.add_argument("--hash-rate", type=float, default=0.5)

    # activations and loss
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--loss-function", type=str,
                        default="mse")  # or bce or wbce
    parser.add_argument(
        "--loss-weights", type=dash_separated_floats, default="1.0-1.0"
    )  # for wbce
    parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
    parser.add_argument("--round-targets", type=bool, default=False)
    # data
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument(
        "--data-generation", type=str, default="dataset"
    )  # synthetic or dataset
    parser.add_argument(
        "--rand-data-dist", type=str, default="uniform"
    )  # uniform or gaussian
    parser.add_argument("--rand-data-min", type=float, default=0)
    parser.add_argument("--rand-data-max", type=float, default=1)
    parser.add_argument("--rand-data-mu", type=float, default=-1)
    parser.add_argument("--rand-data-sigma", type=float, default=1)
    parser.add_argument("--data-trace-file", type=str,
                        default="./input/dist_emb_j.log")
    parser.add_argument("--data-set", type=str,
                        default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str,
                        default="../criteo/kaggle_processed_sparse.bin")
    parser.add_argument("--data-randomize", type=str,
                        default="total")  # or day or none
    parser.add_argument("--data-trace-enable-padding",
                        type=bool, default=False)
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate",
                        type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed",
                        type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--memory-map", action="store_true", default=False)
    # training
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument(
        "--dataset-multiprocessing",
        action="store_true",
        default=False,
        help="The Kaggle dataset can be multiprocessed in an environment \
                        with more than 7 CPU cores and more than 20 GB of memory. \n \
                        The Terabyte dataset can be multiprocessed in an environment \
                        with more than 24 CPU cores and at least 1 TB of memory.",
    )
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # quantize
    parser.add_argument("--quantize-mlp-with-bit", type=int, default=32)
    parser.add_argument("--quantize-emb-with-bit", type=int, default=32)
    # onnx
    parser.add_argument("--save-onnx", action="store_true", default=False)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # distributed
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dist-backend", type=str, default="")
    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    parser.add_argument("--test-num-workers", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=False)
    parser.add_argument("--print-wall-time",
                        action="store_true", default=False)
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--enable-profiling",
                        action="store_true", default=False)
    parser.add_argument("--plot-compute-graph",
                        action="store_true", default=False)
    parser.add_argument("--tensor-board-filename",
                        type=str, default="run_kaggle_pt")
    # store/load model
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    # LR policy
    parser.add_argument("--lr-num-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-start-step", type=int, default=0)
    parser.add_argument("--lr-num-decay-steps", type=int, default=0)
    parser.add_argument("--notinsert-test", action="store_true", default=False)
    parser.add_argument("--ada-flag", action="store_true", default=False)

    parser.add_argument("--cat-path", type=str,
                        default="../criteo_24days/sparse")
    parser.add_argument("--dense-path", type=str,
                        default="../criteo_24days/dense")
    parser.add_argument("--label-path", type=str,
                        default="../criteo_24days/label")
    parser.add_argument("--count-path", type=str,
                        default="../criteo_24days/processed_count.bin")

    parser.add_argument("--sketch-threshold", type=int, default=500)
    parser.add_argument("--use-huffman-coding", action="store_true", default=False)
    parser.add_argument("--use-hashing-trick", action="store_true", default=False)
    parser.add_argument("--use-hash-embedding", action="store_true", default=False)
    parser.add_argument("--num-huffman-digits", type=int, default=2)
    parser.add_argument("--huffman-coding-tensors-path", type=str, default="")
    parser.add_argument("--is-coding-reversed", action="store_true", default=False)
    parser.add_argument("--reverse-and-shift-huffman-coding", action="store_true", default=False)
    parser.add_argument("--use-adaptive-encoding", action="store_true", default=False)
    parser.add_argument("--num-rand-rows", type=int, default=0)
    parser.add_argument("--num-hashes", type=int, default=2)
    parser.add_argument("--num-long-tail-hashes", type=int, default=10)
    parser.add_argument("--num-short-head-hashes", type=int, default=10)
    parser.add_argument("--long-tail-memory-ratio", type=float, default=0.5)
    parser.add_argument("--num-buckets", type=int, default=40)
    parser.add_argument("--frequency-percentile", type=float, default=0.05)
    parser.add_argument("--compression-ratio", type=int, default=10)
    parser.add_argument("--sketching-size", type=int, default=10)
    # CLS Cold Start 모듈
    parser.add_argument("--use-cls", action="store_true", default=False,
                        help="CLS 기반 Cold Start 완화 모듈 활성화")
    parser.add_argument("--cls-alpha", type=float, default=None,
                        help="임베딩 이전 가중치 α 고정값 (지정 시 alpha_min=alpha_max=해당값)")
    parser.add_argument("--cls-alpha-min", type=float, default=0.1,
                        help="동적 alpha 최솟값 (학습 초반, 배치 0)")
    parser.add_argument("--cls-alpha-max", type=float, default=0.9,
                        help="동적 alpha 최댓값 (학습 후반, 배치 total_batches)")
    parser.add_argument("--cls-surge-k", type=int, default=24,
                        help="전환 특성에 일시적으로 부여할 해시 함수 수 (기본 k보다 커야 효과적)")
    parser.add_argument("--cls-decay-freq", type=int, default=10000,
                        help="Dynamic SMED: 몇 배치마다 short_head_indices_set decay를 적용할지")
    parser.add_argument("--cls-decay-rate", type=float, default=0.5,
                        help="Dynamic SMED: 상위 몇 %를 유지할지 (0.5=상위 50% 유지, 하위 50% 제거)")


    global args
    global nbatches
    global nbatches_test
    global writer
    args = parser.parse_args()

    if args.hash_flag:
        print(f"hash_flag: {args.hash_flag}")
    elif args.use_huffman_coding:
        print(f"use_huffman_coding: {args.use_huffman_coding}")
    elif args.sketch_flag:
        print(f"sketch_flag: {args.sketch_flag}")
    elif args.use_adaptive_encoding:
        print(f"use_adaptive_encoding: {args.use_adaptive_encoding}")
    elif args.use_hash_embedding:
        print(f"use_hash_embedding: {args.use_hash_embedding}")
    else:
        print("default setting")

    if args.is_coding_reversed:
        print("is_coding_reversed")

    if args.dataset_multiprocessing:
        assert float(sys.version[:3]) > 3.7, (
            "The dataset_multiprocessing "
            + "flag is susceptible to a bug in Python 3.7 and under. "
            + "https://github.com/facebookresearch/dlrm/issues/172"
        )

    if args.weighted_pooling is not None:
        if args.qr_flag:
            sys.exit(
                "ERROR: quotient remainder with weighted pooling is not supported")
        if args.md_flag:
            sys.exit(
                "ERROR: mixed dimensions with weighted pooling is not supported")
    if args.quantize_emb_with_bit in [4, 8]:
        if args.qr_flag:
            sys.exit(
                "ERROR: 4 and 8-bit quantization with quotient remainder is not supported"
            )
        if args.md_flag:
            sys.exit(
                "ERROR: 4 and 8-bit quantization with mixed dimensions is not supported"
            )
        if args.use_gpu:
            sys.exit("ERROR: 4 and 8-bit quantization on GPU is not supported")

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.numpy_rand_seed)

    if args.test_mini_batch_size < 0:
        # if the parameter is not set, use the training batch size
        args.test_mini_batch_size = args.mini_batch_size
    if args.test_num_workers < 0:
        # if the parameter is not set, use the same parameter for training
        args.test_num_workers = args.num_workers

    use_gpu = args.use_gpu and torch.cuda.is_available()
    print(f"availble: {torch.cuda.is_available()}")

    if use_gpu:
        torch.cuda.manual_seed_all(args.numpy_rand_seed)
        torch.backends.cudnn.deterministic = True
        ngpus = torch.cuda.device_count()
        device = torch.device("cuda", 0)
        print("Using {} GPU(s)...".format(ngpus))
    else:
        device = torch.device("cpu")
        print("Using CPU...")

    ### prepare training data ###
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    # input data

    if args.data_generation == "dataset":
        train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        nbatches_test = len(test_ld)
        ln_emb = train_data.counts

        capacity = get_capacity(args.compression_ratio, ln_emb)
        print(f"compression_ratio: {args.compression_ratio} capacity: {capacity}")
        
        online_frequency_checker = None
        selected_ln_emb_cum_offsets = None
        compressed_table_mask = ln_emb > capacity
        long_tail_hash = None
        short_head_hash = None
        surge_long_tail_hash = None
        short_head_indices_set = set()

        if args.use_adaptive_encoding:
            selected_ln_emb = ln_emb[compressed_table_mask]
            selected_ln_emb_tensor = torch.from_numpy(selected_ln_emb)
            cumulative_sum = torch.cumsum(selected_ln_emb_tensor, dim=0)
            selected_ln_emb_cum_offsets = torch.cat((torch.tensor([0]), cumulative_sum[:-1]))
            selected_ln_emb_cum_offsets = selected_ln_emb_cum_offsets.to(device)
            total_unique_indices = sum(selected_ln_emb)

            online_frequency_checker = ae.OnlineFrequencyChecker(args.sketching_size)
            num_buckets_long_tail = int(capacity * args.long_tail_memory_ratio)
            num_buckets_short_head = capacity - num_buckets_long_tail
            long_tail_hash = HashEmbedding(total_unique_indices, args.arch_sparse_feature_size, num_buckets=num_buckets_long_tail, num_hashes=args.num_long_tail_hashes, append_weight=False, device=device)
            short_head_hash = None if num_buckets_short_head <= 0 else HashEmbedding(total_unique_indices, args.arch_sparse_feature_size, num_buckets=num_buckets_short_head, num_hashes=args.num_short_head_hashes, append_weight=False, offsets=num_buckets_long_tail, device=device)
            # surge-k: 전환 특성에 더 많은 해시 함수 적용 (같은 lt 버킷 범위, 다른 hash 함수)
            surge_k = args.cls_surge_k if args.use_cls else args.num_long_tail_hashes
            surge_long_tail_hash = HashEmbedding(total_unique_indices, args.arch_sparse_feature_size, num_buckets=num_buckets_long_tail, num_hashes=surge_k, append_weight=False, device=device) if (args.use_cls and surge_k > args.num_long_tail_hashes) else None

        print(f"ln_emb: {ln_emb}")
        hash_rate = 0
        # enforce maximum limit on number of vectors per embedding
        if args.max_ind_range > 0:
            ln_emb = np.array(
                list(
                    map(
                        lambda x: x if x < args.max_ind_range else args.max_ind_range,
                        ln_emb,
                    )
                )
            )
        else:
            ln_emb = np.array(ln_emb)
        if args.data_set == 'kaggle' or args.data_set == 'terabyte':
            m_den = 13
            ln_bot[0] = 13
        if args.data_set == 'avazu' or args.data_set == 'kdd12':
            m_den = 0
            ln_bot[0] = 0

    """
    else:
        # input and target at random
        ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
        m_den = ln_bot[0]
        train_data, train_ld, test_data, test_ld = dp.make_random_data_and_loader(
            args, ln_emb, m_den
        )
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        nbatches_test = len(test_ld)
    """

    args.ln_emb = ln_emb.tolist()

    ### parse command line arguments ###
    m_spa = args.arch_sparse_feature_size
    hotn = 0
    if args.sketch_flag:
        totn = sum(ln_emb)
        hotn = int(totn * args.compress_rate * (1 - args.hash_rate)
                   * (m_spa * 4 / (m_spa * 4 + 48)))
        hash_rate = args.compress_rate * args.hash_rate
        print(f"hash_rate: {hash_rate}, hotn: {hotn}")
    ln_emb = np.asarray(ln_emb)

    hash_embeddings = []
    if args.use_hash_embedding:
        for i in range(ln_emb.size):
            if ln_emb[i] > capacity:
                hash_embeddings.append(HashEmbedding(ln_emb[i], m_spa, num_buckets=capacity, num_hashes=args.num_hashes, append_weight=False))
            else:
                hash_embeddings.append(None)

    if args.data_set != 'avazu' and args.data_set != 'kdd12':
        num_fea = ln_emb.size + 1  # num sparse + num dense features
    else:
        num_fea = ln_emb.size
    if args.data_set == 'avazu' or args.data_set == 'kdd12':
        m_den_out = 0
    else:
        m_den_out = ln_bot[ln_bot.size - 1]
    if args.arch_interaction_op == "dot":
        # approach 1: all
        # num_int = num_fea * num_fea + m_den_out
        # approach 2: unique
        if args.arch_interaction_itself:
            num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
        else:
            num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    elif args.arch_interaction_op == "cat":
        num_int = num_fea * m_den_out
    else:
        sys.exit(
            "ERROR: --arch-interaction-op="
            + args.arch_interaction_op
            + " is not supported"
        )
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    print(arch_mlp_top_adjusted)
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    # sanity check: feature sizes and mlp dimensions must match
    if m_den != ln_bot[0]:
        sys.exit(
            "ERROR: arch-dense-feature-size "
            + str(m_den)
            + " does not match first dim of bottom mlp "
            + str(ln_bot[0])
        )

    if args.qr_flag and args.data_set != 'avazu' and args.data_set != 'kdd12':
        if args.qr_operation == "concat" and 2 * m_spa != m_den_out:
            sys.exit(
                "ERROR: 2 arch-sparse-feature-size "
                + str(2 * m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
                + " (note that the last dim of bottom mlp must be 2x the embedding dim)"
            )
        if args.qr_operation != "concat" and m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    else:
        if m_spa != m_den_out and args.data_set != 'avazu' and args.data_set != 'kdd12':
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    if num_int != ln_top[0]:
        sys.exit(
            "ERROR: # of feature interactions "
            + str(num_int)
            + " does not match first dimension of top mlp "
            + str(ln_top[0])
        )

    # assign mixed dimensions if applicable
    if args.md_flag:
        l = 0.0001
        r = 0.5
        while r - l > 0.0001:
            mid = (l + r) / 2
            m_spa_ = md_solver(
                torch.tensor(ln_emb),
                mid,  # alpha
                d0=m_spa,
                round_dim=args.md_round_dims,
            ).tolist()
            cr = sum(m_spa_ * ln_emb) / (np.sum(ln_emb) * m_spa)
            if cr > args.compress_rate:
                l = mid
            else:
                r = mid
        m_spa = md_solver(
            torch.tensor(ln_emb),
            r,  # alpha
            d0=m_spa,
            round_dim=args.md_round_dims,
        ).tolist()

    # test prints (model arch)
    if args.debug_mode:
        print("model arch:")
        print(
            "mlp top arch "
            + str(ln_top.size - 1)
            + " layers, with input to output dimensions:"
        )
        print(ln_top)
        print("# of interactions")
        print(num_int)
        print(
            "mlp bot arch "
            + str(ln_bot.size - 1)
            + " layers, with input to output dimensions:"
        )
        print(ln_bot)
        print("# of features (sparse and dense)")
        print(num_fea)
        print("dense feature size")
        print(m_den)
        print("sparse feature size")
        print(m_spa)
        print(
            "# of embeddings (= # of sparse features) "
            + str(ln_emb.size)
            + ", with dimensions "
            + str(m_spa)
            + "x:"
        )
        print(ln_emb)

        print("data (inputs and targets):")
        for j, inputBatch in enumerate(train_ld):
            X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)

            torch.set_printoptions(precision=4)
            # early exit if nbatches was set by the user and has been exceeded
            if nbatches > 0 and j >= nbatches:
                break
            print("mini-batch: %d" % j)
            print(X.detach().cpu())
            # transform offsets to lengths when printing
            print(
                torch.IntTensor(
                    [
                        np.diff(
                            S_o.detach().cpu().tolist() + list(lS_i[i].shape)
                        ).tolist()
                        for i, S_o in enumerate(lS_o)
                    ]
                )
            )
            print([S_i.detach().cpu() for S_i in lS_i])
            print(T.detach().cpu())

    global ndevices
    ndevices = min(ngpus, args.mini_batch_size, num_fea - 1) if use_gpu else -1

    huffman_coding_dicts = [{} for _ in range(ln_emb.size)]
    huffman_coding_tensors = []
    max_lens = [0 for _ in range(ln_emb.size)]

    if args.use_huffman_coding:
        with open(args.huffman_coding_tensors_path, 'rb') as file:
            huffman_coding_tensors = pickle.load(file)
            max_lens = pickle.load(file)
            print(f"max_lens: {max_lens}")

    ### construct the neural network specified above ###
    # WARNING: to obtain exactly the same initialization for
    # the weights we need to start from the same random seed.
    # np.random.seed(args.numpy_rand_seed)
    lib = None
    if args.sketch_flag:
        print("build sklib.so begin")
        os.system("g++ -fPIC -shared -o tricks/sklib.so -g -rdynamic -mavx2 -mbmi2 --std=c++17 -O3 -fopenmp tricks/sketch.cpp")
        print("build sklib.so end")
        print("create DLL sklib.so begin")
        lib = ctypes.CDLL('./tricks/sklib.so')
        print("create DLL sklib.so end")

    global dlrm
    dlrm = DLRM_Net(
        args.compress_rate,
        args.ada_flag,
        lib,
        hotn,
        device,
        args.sketch_flag,
        hash_rate,
        m_spa,
        ln_emb,
        ln_bot,
        ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        sync_dense_params=args.sync_dense_params,
        loss_threshold=args.loss_threshold,
        ndevices=ndevices,
        qr_flag=args.qr_flag,
        qr_operation=args.qr_operation,
        qr_collisions=args.qr_collisions,
        qr_threshold=args.qr_threshold,
        md_flag=args.md_flag,
        md_threshold=args.md_threshold,
        weighted_pooling=args.weighted_pooling,
        loss_function=args.loss_function,
        max_lens=max_lens,
        use_huffman_coding=args.use_huffman_coding,
        capacity=capacity,
        use_hashing_trick=args.use_hashing_trick,
        num_huffman_digits=args.num_huffman_digits,
        use_adaptive_encoding=args.use_adaptive_encoding,
        use_hash_embedding=args.use_hash_embedding,
        num_buckets=args.num_buckets
    )

    # -----------------------------------------------------------------
    # CLS 모듈 초기화 (dlrm.emb_l 생성 후)
    # -----------------------------------------------------------------
    transfer_module = None
    dynamic_lr = None
    if args.use_adaptive_encoding and args.use_cls:
        if long_tail_hash is not None and selected_ln_emb_cum_offsets is not None:
            # --cls-alpha 고정값이 지정된 경우 alpha_min=alpha_max로 동작
            if args.cls_alpha is not None:
                cls_alpha_min = args.cls_alpha
                cls_alpha_max = args.cls_alpha
            else:
                cls_alpha_min = args.cls_alpha_min
                cls_alpha_max = args.cls_alpha_max
            transfer_module = TransferModule(
                long_tail_hash=long_tail_hash,
                short_head_hash=short_head_hash,
                emb_l=dlrm.emb_l,
                compressed_table_mask=compressed_table_mask,
                ln_emb=ln_emb,
                selected_ln_emb_cum_offsets=selected_ln_emb_cum_offsets,
                alpha_min=cls_alpha_min,
                alpha_max=cls_alpha_max,
                total_batches=nbatches,
            )
            dynamic_lr = DynamicLR(base_lr=args.learning_rate)
            print(
                f"[CLS] TransferModule 초기화 완료 "
                f"(alpha_min={cls_alpha_min}, alpha_max={cls_alpha_max}, "
                f"total_batches={nbatches}, "
                f"압축 테이블 수={len(transfer_module.compressed_table_indices)})"
            )
            print(f"[CLS] {dynamic_lr}")
        else:
            print("[CLS] --use-cls 지정됐지만 adaptive encoding이 비활성 상태 → CLS 스킵")
    # -----------------------------------------------------------------

    # test prints
    if args.debug_mode:
        print("initial parameters (weights and bias):")
        for param in dlrm.parameters():
            print(param.detach().cpu().numpy())

    if use_gpu:
        # Custom Model-Data Parallel
        # the mlps are replicated and use data parallelism, while
        # the embeddings are distributed and use model parallelism
        dlrm = dlrm.to(device)  # .cuda()
        if dlrm.ndevices > 1:
            dlrm.emb_l, dlrm.v_W_l = dlrm.create_emb(
                m_spa, ln_emb, args.weighted_pooling, max_lens, args.use_huffman_coding, args.use_hashing_trick, args.num_huffman_digits, args.use_adaptive_encoding, args.use_hash_embedding, args.num_buckets
            )
        else:
            if dlrm.weighted_pooling == "fixed":
                for k, w in enumerate(dlrm.v_W_l):
                    dlrm.v_W_l[k] = w.cuda()

    if not args.inference_only:
        if use_gpu and args.optimizer in ["rwsadagrad", "adagrad"]:
            sys.exit("GPU version of Adagrad is not supported by PyTorch.")
        # specify the optimizer algorithm
        opts = {
            "sgd": torch.optim.SGD,
            # "rwsadagrad": RowWiseSparseAdagrad.RWSAdagrad,
            "adagrad": torch.optim.Adagrad,
        }

        parameters = (
            dlrm.parameters()
        )
        optimizer = opts[args.optimizer](parameters, lr=args.learning_rate)
        lr_scheduler = LRPolicyScheduler(
            optimizer,
            args.lr_num_warmup_steps,
            args.lr_decay_start_step,
            args.lr_num_decay_steps,
        )

    ### main loop ###

    # training or inference
    best_metric = [0, 0] # [best_acc_test, best_auc_test]
    skip_upto_epoch = 0
    skip_upto_batch = 0
    total_time = 0
    total_loss = 0
    total_iter = 0
    total_samp = 0
    internal_total_time = 0
    external_total_time = 0

    # Load model is specified
    if not (args.load_model == ""):
        print("Loading saved model {}".format(args.load_model))
        if use_gpu:
            if dlrm.ndevices > 1:
                # NOTE: when targeting inference on multiple GPUs,
                # load the model as is on CPU or GPU, with the move
                # to multiple GPUs to be done in parallel_forward
                ld_model = torch.load(args.load_model)
            else:
                # NOTE: when targeting inference on single GPU,
                # note that the call to .to(device) has already happened
                ld_model = torch.load(
                    args.load_model,
                    map_location=torch.device("cuda")
                    # map_location=lambda storage, loc: storage.cuda(0)
                )
        else:
            # when targeting inference on CPU
            ld_model = torch.load(
                args.load_model, map_location=torch.device("cpu"))
        dlrm.load_state_dict(ld_model["state_dict"])
        if args.sketch_flag:
            dlrm.lib.update()
        ld_j = ld_model["iter"]
        ld_k = ld_model["epoch"]
        ld_nepochs = ld_model["nepochs"]
        ld_nbatches = ld_model["nbatches"]
        ld_nbatches_test = ld_model["nbatches_test"]
        ld_train_loss = ld_model["train_loss"]
        ld_total_loss = ld_model["total_loss"]
        ld_acc_test = ld_model["test_acc"]
        if not args.inference_only:
            optimizer.load_state_dict(ld_model["opt_state_dict"])
            best_metric[0] = ld_acc_test
            total_loss = ld_total_loss
            skip_upto_epoch = ld_k  # epochs
            skip_upto_batch = ld_j  # batches
        else:
            args.print_freq = ld_nbatches
            args.test_freq = 0

        print(
            "Saved at: epoch = {:d}/{:d}, batch = {:d}/{:d}, ntbatch = {:d}".format(
                ld_k, ld_nepochs, ld_j, ld_nbatches, ld_nbatches_test
            )
        )
        print(
            "Training state: loss = {:.6f}".format(
                ld_train_loss,
            )
        )

        print("Testing state: accuracy = {:3.3f} %".format(
            ld_acc_test * 100))

    if args.inference_only:
        # Currently only dynamic quantization with INT8 and FP16 weights are
        # supported for MLPs and INT4 and INT8 weights for EmbeddingBag
        # post-training quantization during the inference.
        # By default we don't do the quantization: quantize_{mlp,emb}_with_bit == 32 (FP32)
        assert args.quantize_mlp_with_bit in [
            8,
            16,
            32,
        ], "only support 8/16/32-bit but got {}".format(args.quantize_mlp_with_bit)
        assert args.quantize_emb_with_bit in [
            4,
            8,
            32,
        ], "only support 4/8/32-bit but got {}".format(args.quantize_emb_with_bit)
        if args.quantize_mlp_with_bit != 32:
            if args.quantize_mlp_with_bit in [8]:
                quantize_dtype = torch.qint8
            else:
                quantize_dtype = torch.float16
            dlrm = torch.quantization.quantize_dynamic(
                dlrm, {torch.nn.Linear}, quantize_dtype
            )
        if args.quantize_emb_with_bit != 32:
            dlrm.quantize_embedding(args.quantize_emb_with_bit)
            # print(dlrm)

    print("time/loss/accuracy (if enabled):")

    tb_file = "./" + args.tensor_board_filename
    writer = SummaryWriter(tb_file)

    time_start = time.time()

    with torch.autograd.profiler.profile(
        args.enable_profiling, use_cuda=use_gpu, record_shapes=True
    ) as prof:
        if not args.inference_only:
            k = 0
            total_time_begin = 0
            while k < args.nepochs:

                if k < skip_upto_epoch:
                    continue

                num = 0
                t1 = 0
                t2 = 0

                train_start_time = time.time()

                for j, inputBatch in enumerate(train_ld):
                    if j == 0 and args.save_onnx:
                        X_onnx, lS_o_onnx, lS_i_onnx, _, _, _ = unpack_batch(
                            inputBatch)

                    if j < skip_upto_batch:
                        continue
                    X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)
                    t1 = time_wrap(use_gpu)

                    # early exit if nbatches was set by the user and has been exceeded
                    if nbatches > 0 and j >= nbatches:
                        break

                    # = args.mini_batch_size except maybe for last
                    mbs = T.shape[0]
                    
                    internal_train_timer = [0]
                    external_train_timer = time.time()

                    # forward pass
                    Z = dlrm_wrap(
                        X,
                        lS_o,
                        lS_i,
                        use_gpu,
                        device,
                        ndevices=ndevices,
                        test=False,
                        sk_flag=args.sketch_flag,
                        use_huffman_coding=args.use_huffman_coding,
                        huffman_coding_tensors=huffman_coding_tensors,
                        max_lens=max_lens,
                        ln_emb=ln_emb,
                        capacity=capacity,
                        use_hashing_trick=args.use_hashing_trick,
                        is_reversed=args.is_coding_reversed,
                        reverse_and_shift_huffman_coding=args.reverse_and_shift_huffman_coding,
                        num_huffman_digits=args.num_huffman_digits,
                        use_adaptive_encoding=args.use_adaptive_encoding,
                        online_frequency_checker=online_frequency_checker,
                        compressed_table_mask=compressed_table_mask,
                        selected_ln_emb_cum_offsets=selected_ln_emb_cum_offsets,
                        use_hash_embedding=args.use_hash_embedding,
                        hash_embeddings=hash_embeddings,
                        num_buckets=args.num_buckets,
                        short_head_hash=short_head_hash,
                        long_tail_hash=long_tail_hash,
                        iteration_index=j,
                        frequency_percentile=args.frequency_percentile,
                        short_head_indices_set=short_head_indices_set,
                        train_time=internal_train_timer,
                        transfer_module=transfer_module,
                        surge_long_tail_hash=surge_long_tail_hash,
                    )

                    # Dynamic SMED decay: short_head_indices_set에서
                    # 저빈도 항목을 주기적으로 제거 → 재전환 유도
                    if (
                        args.use_cls
                        and online_frequency_checker is not None
                        and transfer_module is not None
                    ):
                        decayed = online_frequency_checker.apply_decay(
                            short_head_indices_set=short_head_indices_set,
                            decay_rate=args.cls_decay_rate,
                            current_batch=j,
                            decay_freq=args.cls_decay_freq,
                        )
                        if decayed:
                            # _prev_short_head_set을 post-decay 상태로 갱신
                            # → 다음 배치에서 제거된 항목이 재진입하면 새 전환으로 감지
                            transfer_module.sync_prev_set(short_head_indices_set)
                            print(
                                f"[CLS] Dynamic SMED decay 적용 "
                                f"(저빈도로 내려간 특성: {len(decayed)}개, "
                                f"decay_rate={args.cls_decay_rate}, "
                                f"batch={j})"
                            )

                    if (j + 1) % 10000 == 0:
                        train_end_time = time.time()
                        train_time_duration = (train_end_time - train_start_time)
                        print(f"train batch {j - 9999}-{j} takes {train_time_duration} seconds")
                        train_start_time = time.time()

                    # loss
                    E = loss_fn_wrap(Z, T, use_gpu, device)

                    # compute loss and accuracy
                    L = E.detach().cpu().numpy()  # numpy array
                    # training accuracy is not disabled
                    # S = Z.detach().cpu().numpy()  # numpy array
                    # T = T.detach().cpu().numpy()  # numpy array

                    # # print("res: ", S)

                    # # print("j, train: BCE ", j, L)

                    # mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
                    # A = np.sum((np.round(S, 0) == T).astype(np.uint8))

                    with record_function("DLRM backward"):
                        # scaled error gradient propagation
                        # (where we do not accumulate gradients across mini-batches)
                        optimizer.zero_grad()
                        # backward pass
                        E.backward()

                        if args.sketch_flag:
                            dlrm.insert_grad(lS_i)
                        if args.ada_flag:
                            dlrm.insert_adagrad(lS_i)
                        # optimizer
                        optimizer.step()
                        lr_scheduler.step()
                    
                    internal_time_diff = time.time() - internal_train_timer[0]
                    internal_total_time += internal_time_diff
                    external_time_diff = time.time() - external_train_timer
                    external_total_time += external_time_diff

                    t2 = time_wrap(use_gpu)
                    total_time += t2 - t1

                    total_loss += L * mbs
                    total_iter += 1
                    total_samp += mbs

                    should_print = ((j + 1) % args.print_freq == 0) or (
                        j + 1 == nbatches
                    ) or (j <= 100)
                    should_test = (
                        (args.test_freq > 0)
                        and (args.data_generation in ["dataset", "random"])
                        and (((j + 1) % args.test_freq == 0) or (j + 1 == nbatches))
                    )

                    # print time, loss and accuracy
                    if should_print or should_test:
                        gT = 1000.0 * total_time / total_iter if args.print_time else -1
                        total_time = 0

                        train_loss = total_loss / total_samp
                        total_loss = 0

                        str_run_type = (
                            "inference" if args.inference_only else "training"
                        )

                        wall_time = ""
                        if args.print_wall_time:
                            wall_time = " ({})".format(time.strftime("%H:%M"))

                        print(
                            "Finished {} it {}/{} of epoch {}, {:.2f} ms/it,".format(
                                str_run_type, j + 1, nbatches, k, gT,
                            )
                            + " loss {:.6f}".format(train_loss)
                            + wall_time,
                            flush=True,
                        )

                        log_iter = nbatches * k + j + 1
                        writer.add_scalar("Train/Loss", train_loss, log_iter)

                        total_iter = 0
                        total_samp = 0
                        reset_sketch_time()

                    # testing

                    if should_test:
                        epoch_num_float = (j + 1) / len(train_ld) + k + 1

                        print(
                            "Testing at - {}/{} of epoch {},".format(
                                j + 1, nbatches, k)
                        )
                        model_metrics_dict, is_best_acc, _ = inference(
                            args,
                            dlrm,
                            best_metric,
                            test_ld,
                            device,
                            use_gpu,
                            log_iter,
                            args.notinsert_test,
                            args.sketch_flag,
                            args.use_huffman_coding,
                            huffman_coding_tensors,
                            max_lens,
                            ln_emb,
                            capacity,
                            args.use_hashing_trick,
                            args.is_coding_reversed,
                            args.reverse_and_shift_huffman_coding,
                            args.num_huffman_digits,
                            args.use_adaptive_encoding,
                            online_frequency_checker,
                            compressed_table_mask,
                            selected_ln_emb_cum_offsets,
                            args.use_hash_embedding,
                            hash_embeddings,
                            args.num_buckets,
                            short_head_hash,
                            long_tail_hash,
                            j,
                            args.frequency_percentile,
                            short_head_indices_set
                        )

                        if (
                            is_best_acc
                            and not (args.save_model == "")
                            and not args.inference_only
                        ):
                            model_metrics_dict["epoch"] = k
                            model_metrics_dict["iter"] = j + 1
                            model_metrics_dict["train_loss"] = train_loss
                            model_metrics_dict["total_loss"] = total_loss
                            model_metrics_dict[
                                "opt_state_dict"
                            ] = optimizer.state_dict()
                            print("Saving model to {}".format(args.save_model))
                            torch.save(model_metrics_dict, args.save_model)

                        # Uncomment the line below to print out the total time with overhead
                        # print("Total test time for this group: {}" \
                        # .format(time_wrap(use_gpu) - accum_test_time_begin))

                k += 1  # nepochs
        else:
            print("Testing for inference only")
            inference(
                args,
                dlrm,
                best_metric,
                test_ld,
                device,
                use_gpu,
                log_iter=-1,
                test=False,
                sk_flag=False,
                use_huffman_coding=args.use_huffman_coding,
                huffman_coding_tensors=huffman_coding_tensors,
                max_lens=max_lens,
                ln_emb=ln_emb,
                capacity=capacity,
                use_hashing_trick=args.use_hashing_trick,
                is_coding_reversed=args.is_coding_reversed,
                reverse_and_shift_huffman_coding=args.reverse_and_shift_huffman_coding,
                num_huffman_digits=args.num_huffman_digits,
                use_adaptive_encoding=args.use_adaptive_encoding,
                online_frequency_checker=online_frequency_checker,
                compressed_table_mask=compressed_table_mask,
                selected_ln_emb_cum_offsets=selected_ln_emb_cum_offsets,
                use_hash_embedding=args.use_hash_embedding,
                hash_embeddings=hash_embeddings,
                num_buckets=args.num_buckets,
                short_head_hash=short_head_hash,
                long_tail_hash=long_tail_hash,
                iteration_index=-1,
                frequency_percentile=args.frequency_percentile,
                short_head_indices_set=short_head_indices_set,
                transfer_module=None,
                surge_long_tail_hash=None,
            )

    # profiling
    if args.enable_profiling:
        time_stamp = str(datetime.datetime.now()).replace(" ", "_")
        with open("dlrm_s_pytorch" + time_stamp + "_shape.prof", "w") as prof_f:
            prof_f.write(
                prof.key_averages(group_by_input_shape=True).table(
                    sort_by="self_cpu_time_total"
                )
            )
        with open("dlrm_s_pytorch" + time_stamp + "_total.prof", "w") as prof_f:
            prof_f.write(prof.key_averages().table(
                sort_by="self_cpu_time_total"))
        prof.export_chrome_trace("dlrm_s_pytorch" + time_stamp + ".json")
        # print(prof.key_averages().table(sort_by="cpu_time_total"))

    # plot compute graph
    if args.plot_compute_graph:
        sys.exit(
            "ERROR: Please install pytorchviz package in order to use the"
            + " visualization. Then, uncomment its import above as well as"
            + " three lines below and run the code again."
        )
        # V = Z.mean() if args.inference_only else E
        # dot = make_dot(V, params=dict(dlrm.named_parameters()))
        # dot.render('dlrm_s_pytorch_graph') # write .pdf file

    # test prints
    if not args.inference_only and args.debug_mode:
        print("updated parameters (weights and bias):")
        for param in dlrm.parameters():
            print(param.detach().cpu().numpy())

    # export the model in onnx
    if args.save_onnx:
        """
        # workaround 1: tensor -> list
        if torch.is_tensor(lS_i_onnx):
            lS_i_onnx = [lS_i_onnx[j] for j in range(len(lS_i_onnx))]
        # workaound 2: list -> tensor
        lS_i_onnx = torch.stack(lS_i_onnx)
        """
        # debug prints
        # print("inputs", X_onnx, lS_o_onnx, lS_i_onnx)
        # print("output", dlrm_wrap(X_onnx, lS_o_onnx, lS_i_onnx, use_gpu, device))
        dlrm_pytorch_onnx_file = "dlrm_s_pytorch.onnx"
        batch_size = X_onnx.shape[0]
        print("X_onnx.shape", X_onnx.shape)
        if torch.is_tensor(lS_o_onnx):
            print("lS_o_onnx.shape", lS_o_onnx.shape)
        else:
            for oo in lS_o_onnx:
                print("oo.shape", oo.shape)
        if torch.is_tensor(lS_i_onnx):
            print("lS_i_onnx.shape", lS_i_onnx.shape)
        else:
            for ii in lS_i_onnx:
                print("ii.shape", ii.shape)

        # name inputs and outputs
        o_inputs = (
            ["offsets"]
            if torch.is_tensor(lS_o_onnx)
            else ["offsets_" + str(i) for i in range(len(lS_o_onnx))]
        )
        i_inputs = (
            ["indices"]
            if torch.is_tensor(lS_i_onnx)
            else ["indices_" + str(i) for i in range(len(lS_i_onnx))]
        )
        all_inputs = ["dense_x"] + o_inputs + i_inputs
        # debug prints
        print("inputs", all_inputs)

        # create dynamic_axis dictionaries
        do_inputs = (
            [{"offsets": {1: "batch_size"}}]
            if torch.is_tensor(lS_o_onnx)
            else [
                {"offsets_" + str(i): {0: "batch_size"}} for i in range(len(lS_o_onnx))
            ]
        )
        di_inputs = (
            [{"indices": {1: "batch_size"}}]
            if torch.is_tensor(lS_i_onnx)
            else [
                {"indices_" + str(i): {0: "batch_size"}} for i in range(len(lS_i_onnx))
            ]
        )
        dynamic_axes = {"dense_x": {0: "batch_size"},
                        "pred": {0: "batch_size"}}
        for do in do_inputs:
            dynamic_axes.update(do)
        for di in di_inputs:
            dynamic_axes.update(di)
        # debug prints
        print(dynamic_axes)
        # export model
        torch.onnx.export(
            dlrm,
            (X_onnx, lS_o_onnx, lS_i_onnx),
            dlrm_pytorch_onnx_file,
            verbose=True,
            opset_version=11,
            input_names=all_inputs,
            output_names=["pred"],
            dynamic_axes=dynamic_axes,
        )
        # recover the model back
        dlrm_pytorch_onnx = onnx.load("dlrm_s_pytorch.onnx")
        # check the onnx model
        onnx.checker.check_model(dlrm_pytorch_onnx)
    total_time_end = time_wrap(use_gpu)
    time_end = time.time()
    print("total time:", time_end - time_start)
    print("internal total time:", internal_total_time)
    print("external total time:", external_total_time)
    print(f"short_head_indices_set size: {len(short_head_indices_set)}")


if __name__ == "__main__":
    run()
