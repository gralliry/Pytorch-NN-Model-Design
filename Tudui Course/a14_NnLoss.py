# -*- coding: utf-8 -*-
# @Time    : 2023/11/17 15:04
# @Author  : Liang Jinaye
# @File    : 14_NnLoss.py
# @Description :
import torch
from torch import nn

gbinp = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

gbinp = torch.reshape(gbinp, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

# 均差值

loss = nn.L1Loss(reduction="sum")
result = loss(gbinp, targets)

print(result)  # 0.6667

# 平方根
loss_mse = nn.MSELoss()
result_mse = loss_mse(gbinp, targets)

print(result_mse)  # 1.333

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)
