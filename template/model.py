#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :

import torch
from torch import nn
import torch.nn.functional as f
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        ...
        return x
