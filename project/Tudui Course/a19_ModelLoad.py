# -*- coding: utf-8 -*-
# @Time    : 2023/11/17 18:50
# @Author  : Liang Jinaye
# @File    : 19_ModelLoad.py
# @Description :
import torch
import torchvision.models
from torch import nn

model1 = torch.load("./parameter/vgg16_method1.pth")
print(model1)

vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("./parameter/vgg16_method2.pth"))

print(vgg16)


# 陷阱
class Tudui(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x


tudui = Tudui()

torch.save(tudui, "./parameter/tudui_method1.pth")
