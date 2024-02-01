# -*- coding: utf-8 -*-
# @Time    : 2023/11/17 21:24
# @Author  : Liang Jinaye
# @File    : 21_Model.py
# @Description :
import torch
from torch import nn


class Tudui(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )
        self.model2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


if __name__ == "__main__":
    tudui = Tudui()
    inp = torch.ones((64, 3, 32, 32))
    output = tudui(inp)
    print(output.shape)
