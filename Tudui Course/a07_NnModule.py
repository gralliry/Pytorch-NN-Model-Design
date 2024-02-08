# -*- coding: utf-8 -*-
# @Time    : 2023/9/10 11:54
# @Author  : Liang Jinaye
# @File    : 07_NnModule.py
# @Description :
import torch
from torch import nn


class Tudui(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(inp):
        # forward函数，要么self和staticmethod不能同时存在
        outp = inp + 1
        return outp


tudui = Tudui()
x = torch.tensor(1.0)
output = tudui(x)
print(output)
