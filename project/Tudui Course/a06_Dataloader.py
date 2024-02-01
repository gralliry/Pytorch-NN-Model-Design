# -*- coding: utf-8 -*-
# @Time    : 2023/9/10 11:01
# @Author  : Liang Jinaye
# @File    : 06_Dataloader.py
# @Description :

import torchvision

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

DATASET_PATH = "E:\\Datasets"

test_data = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# No.1 pic
img, target = test_data[0]
# img.shape -> torch.Size([3, 32, 32])
# target    -> 3

writer = SummaryWriter("./tensorboard/06")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images(f"test_data:{epoch}", imgs, step)
        step += 1

writer.close()
