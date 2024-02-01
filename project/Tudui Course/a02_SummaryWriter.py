# -*- coding: utf-8 -*-
# @Time    : 2023/9/8 16:55
# @Author  : Liang Jinaye
# @File    : 02_SummaryWriter.py
# @Description :

import numpy as np

from torch.utils.tensorboard import SummaryWriter
from PIL import Image

DATASET_PATH = "E:\\Datasets\\hymenoptera_data\\train"

writer = SummaryWriter("./tensorboard/02")
image_path = DATASET_PATH + "\\ants\\5650366_e22b7e1065.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

writer.add_image("test", img_array, 1, dataformats="HWS")
for i in range(100):
    writer.add_scalar("y=2x", 3 * i, i)

# 可视化界面
# tensorboard --logdir=logs --port=6008

writer.close()
