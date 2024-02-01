# -*- coding: utf-8 -*-
# @Time    : 2023/11/16 1:35
# @Author  : Liang Jinaye
# @File    : 10_NnMaxPool.py
# @Description :

# explain dilation https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
# pool

import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

DATASET_PATH = "E:\\Datasets"

dataset = torchvision.datasets.CIFAR10(DATASET_PATH, train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)
gb_inp = torch.tensor([
    [1, 2, 0, 3, 1],
    [0, 1, 2, 3, 1],
    [1, 2, 1, 0, 0],
    [5, 2, 3, 1, 1],
    [2, 1, 0, 1, 1],
], dtype=torch.float32)

gb_inp = torch.reshape(gb_inp, (-1, 1, 5, 5))
print(gb_inp.shape)


class Tudui(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, inp):
        outp = self.maxpool1(inp)
        return outp


tudui = Tudui()

writer = SummaryWriter("./tensorboard/10")
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = tudui(imgs)
    writer.add_images("output", imgs, step)
    step += 1

writer.close()
