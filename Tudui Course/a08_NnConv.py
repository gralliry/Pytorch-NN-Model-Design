# -*- coding: utf-8 -*-
# @Time    : 2023/9/10 13:10
# @Author  : Liang Jinaye
# @File    : 08_NnConv.py
# @Description :

# https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
import torch
import torch.nn.functional as func

# 二维矩阵
inp = torch.tensor([
    [1, 2, 0, 3, 1],
    [0, 1, 2, 3, 1],
    [1, 2, 1, 0, 0],
    [5, 2, 3, 1, 1],
    [2, 1, 0, 1, 1],
])
kernel = torch.tensor([
    [1, 2, 1],
    [0, 1, 0],
    [2, 1, 0]
])

# (1,1,5,5) 4个数字代表4维，分别为为1,1,5,5
inp = torch.reshape(inp, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

# stride 为 卷积核在图像上跳跃的步数
outp1 = func.conv2d(inp, kernel, stride=1)
print(outp1)

outp2 = func.conv2d(inp, kernel, stride=2)
print(outp2)

# padding 为 新增边界宽度，一般为了对齐输入和输出的tensor
# 新加的padding数值一般为 0
outp3 = func.conv2d(inp, kernel, stride=1, padding=1)
print(outp3)

