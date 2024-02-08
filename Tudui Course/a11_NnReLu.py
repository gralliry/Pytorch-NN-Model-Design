# -*- coding: utf-8 -*-
# @Time    : 2023/11/16 14:06
# @Author  : Liang Jinaye
# @File    : 11_NnReLu.py
# @Description :
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

DATASET_PATH = "E:\\Datasets"

gb_inp = torch.tensor([
    [1, -0.5],
    [-1, 3]
])
gb_inp = torch.reshape(gb_inp, (-1, 1, 2, 2))
print(gb_inp.shape)

dataset = torchvision.datasets.CIFAR10(DATASET_PATH, train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.relu1 = nn.ReLU()
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, inp):
        outp = self.sigmoid1(inp)
        return outp


tudui = Tudui()

writer = SummaryWriter("./tensorboard/11")
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output = tudui(imgs)
    writer.add_images("output", output, global_step=step)
    step += 1

writer.close()
