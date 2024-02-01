# -*- coding: utf-8 -*-
# @Time    : 2023/9/10 13:33
# @Author  : Liang Jinaye
# @File    : 09_NnConv2d.py
# @Description :

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

DATASET_PATH = "E:\\Datasets"

dataset = torchvision.datasets.CIFAR10(DATASET_PATH, train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):

    def __init__(self):
        super(self.__class__, self).__init__()
        # 彩色图像，in_channels为3
        # kernel_size如果是int,则自动认为是正边即(3,3)
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


tudui = Tudui()
# print(tudui)

writer = SummaryWriter('./tensorboard/09')
# tensorboard --logdir=tb/09 --port=6008

step = 0
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)
    # torch.Size([64, 6, 30, 30]) -> [xxx, 3, 30, 30]
    # -1 代表自动推断维度
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)

    step += 1

writer.close()
