# -*- coding: utf-8 -*-
# @Time    : 2023/9/10 9:26
# @Author  : Liang Jinaye
# @File    : 05_DatasetTransforms.py
# @Description :

import torchvision

from torch.utils.tensorboard import SummaryWriter

DATASET_PATH = "E:\\Datasets"

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, transform=dataset_transform, download=True)

# test_set[0]      -> (<PIL.image.image image mode=RGB size=32x32 at 0x178329575E0>, 3)
# test_set.classes -> ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# img                      -> <PIL.image.image image mode=RGB size=32x32 at 0x1DE61E08D30>
# target                   -> 3
# test_set.classes[target] -> cat

writer = SummaryWriter(f"./tensorboard/05")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()
