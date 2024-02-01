# -*- coding: utf-8 -*-
# @Time    : 2023/11/16 14:28
# @Author  : Liang Jinaye
# @File    : 12_Linear.py
# @Description :
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

DATASET_PATH = "E:\\Datasets"

dataset = torchvision.datasets.CIFAR10(DATASET_PATH, train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.linear1 = nn.Linear(196608, 10)

    def forward(self, inp):
        outp = self.linear1(inp)
        return outp


tudui = Tudui()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    output = torch.reshape(imgs, (1, 1, 1, -1))
    # 变成一行
    torch.flatten(imgs)
    print(output.shape)
    output = tudui(output)
    print(output.shape)
