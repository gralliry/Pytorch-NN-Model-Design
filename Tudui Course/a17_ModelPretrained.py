# -*- coding: utf-8 -*-
# @Time    : 2023/11/17 17:14
# @Author  : Liang Jinaye
# @File    : 17_ModelPretrained.py
# @Description :

from torch import nn
import torchvision

DATASET_PATH = "E:\\Datasets"

# train_data = torchvision.datasets.ImageNet("./dataset", split="train", download=True,
#                                            transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10(DATASET_PATH, train=False, download=True,
                                          transform=torchvision.transforms.ToTensor())

vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10))
print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
