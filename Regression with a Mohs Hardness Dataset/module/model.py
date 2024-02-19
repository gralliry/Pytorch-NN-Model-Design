#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :

from torch import nn

cfg = [16, 'M', 32, 'M', 64, 'M', 128, 'M', 256, 'M', 512, 'M']


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = self.make_layers(True)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 11, 512 * 11),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512 * 11, 512 * 11),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512 * 11, 1),
            nn.ReLU(True),
        )

    def forward(self, x):
        # [batch_size, 16]
        x = x.unsqueeze(dim=1)
        # [batch_size, 1, 16]
        x = self.features(x)
        # [batch_size, 512, 16]
        x = x.view(x.size(0), -1)
        # [batch_size, 8192]
        x = self.classifier(x)
        # [batch_size, 16]
        x = x.squeeze()
        return x

    @staticmethod
    def make_layers(batch_norm=False):
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
