#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/8 19:26
# @Author  : Gralliry
# @File    : model.py
# @Description :
import torch
from torch import nn

cfg = [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M']


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(10, 80),
            nn.ReLU(True),
            nn.Linear(80, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # [batch_size, 5120]
        x = self.classifier(x).squeeze()
        # [batch_size, 2]
        return x

    def make_layers(self, batch_norm=False):
        layers = []
        in_channels = 1
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool1d(kernel_size=3, stride=1, padding=1)]
            else:
                conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv1d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
