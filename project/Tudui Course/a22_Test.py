# -*- coding: utf-8 -*-
# @Time    : 2023/11/17 23:56
# @Author  : Liang Jinaye
# @File    : a22_Test.py
# @Description :
import torch

outputs = torch.tensor([[0.1, 0.2],
                        [0.3, 0.4]])
print(outputs.argmax(1))
preds = outputs.argmax(1)

targets = torch.tensor([0, 1])
print((preds == targets).sum())
