# -*- coding: utf-8 -*-
# @Time    : 2023/9/10 8:38
# @Author  : Liang Jinaye
# @File    : 04_UsefulTransforms.py
# @Description :
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

#
writer = SummaryWriter(f"./tensorboard/04")
img = Image.open("./image/default.jpg")

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize
trans_normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_normalize = trans_normalize(img_tensor)
writer.add_image("Normalize", img_normalize)

# Resize - 1
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img_tensor)
writer.add_image("Resize - 1", img_resize)

# Resize - Compose - 2
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)  # 这里不用tensor
writer.add_image("Resize - 2", img_resize_2)

# RandomCrop
trans_randomCrop = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_randomCrop, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()
