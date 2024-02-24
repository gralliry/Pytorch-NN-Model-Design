#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :
from torch import nn


class Criterion(nn.Module):
    def __init__(self, ignore_index):
        super(Criterion, self).__init__()
        # 使用KLDivLoss，不需要知道里面的具体细节。
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, x, target):
        return self.criterion(x, target)
