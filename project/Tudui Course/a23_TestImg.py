# -*- coding: utf-8 -*-
# @Time    : 2023/11/18 13:37
# @Author  : Liang Jinaye
# @File    : a23_TestImg.py
# @Description :
import torch
from PIL import Image

import torchvision

image_path = "./image/dog.png"

image = Image.open(image_path)
image = image.convert("RGB")

print(image)

tranform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
image = tranform(image).cuda()
image = torch.reshape(image, (1, 3, 32, 32))
print(image.shape)

model = torch.load("./parameter/a20_train.pth")
print(model)

model.eval()

with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))
