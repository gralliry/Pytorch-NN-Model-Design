# -*- coding: utf-8 -*-
# @Time    : 2023/11/17 18:47
# @Author  : Liang Jinaye
# @File    : 18_ModelSave.py
# @Description :
import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)
# 方法1, 模型结构 + 参数
torch.save(vgg16, "./parameter/vgg16_method1.pth")
# 方法2, 模型参数（推荐）
torch.save(vgg16.state_dict(), "./parameter/vgg16_method2.pth")
# 陷阱1

model = torch.load("parameter/tudui_method1.pth")
print(model)
