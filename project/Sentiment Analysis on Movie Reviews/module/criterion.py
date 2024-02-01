#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/1 1:41
# @Author  : Jianye Liang
# @File    : criterion.py
# @Description :

import torch
from torch import nn


class CustomLoss(nn.Module):
    def __init__(self, class_weight=100, localization_weight=0, confidence_weight=0):
        super(CustomLoss, self).__init__()
        total_weight = class_weight + localization_weight + confidence_weight
        self.class_weight = class_weight / total_weight
        self.class_loss = nn.CrossEntropyLoss()
        #
        self.localization_weight = localization_weight / total_weight
        self.localization_loss = nn.MSELoss()
        #
        self.confidence_weight = confidence_weight / total_weight
        self.confidence_loss = nn.L1Loss()

    def forward(self, predictions, targets):
        # predictions # (batch_size, classes_num)
        # targets     # (batch_size)

        # print(class_loss.item(), localization_loss.item(), confidence_loss.item())

        loss = self.class_weight * self.class_loss(predictions, targets)

        max_values, max_indices = torch.max(predictions, dim=1)
        loss += self.localization_weight * self.localization_loss(max_indices, targets.float())
        loss += self.confidence_weight * self.confidence_loss(max_values, torch.ones_like(targets))
        return loss
