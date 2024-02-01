# -*- coding: utf-8 -*-
# @Time    : 2023/9/10 8:19
# @Author  : Liang Jinaye
# @File    : 03_Transforms.py
# @Description :

from PIL import Image

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

DATASET_PATH = "E:\\Datasets\\hymenoptera_data\\train"

image_path = DATASET_PATH + "\\ants\\0013035.jpg"
img = Image.open(image_path)
#
writer = SummaryWriter(f"./tensorboard/01")

tensor_train = transforms.ToTensor()
tensor_img = tensor_train(img)

writer.add_image("Tensor_img", tensor_img)

writer.close()

# print(tensor_img)
