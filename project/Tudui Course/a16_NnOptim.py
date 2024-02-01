# -*- coding: utf-8 -*-
# @Time    : 2023/11/17 16:33
# @Author  : Liang Jinaye
# @File    : 16_NnOptim.py
# @Description :
import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader

DATASET_PATH = "E:\\Datasets"

dataset = torchvision.datasets.CIFAR10(DATASET_PATH, train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=16)


class Tudui(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


loss = nn.CrossEntropyLoss()

tudui = Tudui().cuda()

optim = torch.optim.SGD(tudui.parameters(), lr=0.001)
for epoch in range(400):
    print(f"No.{epoch}")
    runnning_loss = 0
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs.cuda())
        result_loss = loss(outputs.cuda(), targets.cuda())
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        runnning_loss += result_loss
    print(f"running_loss: {runnning_loss}")
