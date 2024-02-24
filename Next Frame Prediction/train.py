#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/23 18:55
# @Author  : Jianye Liang
# @File    : train.py.py
# @Description :
import os

import cv2
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from dataset import CvImageDataset
from model import UNet

CUDA = True
DICT_PATH = "./checkpoint"


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = CvImageDataset(transforms=transforms.Compose([
        transforms.Resize((64, 64)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为 PyTorch 的 Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到 [-1, 1] 范围
    ]))
    dataloader = DataLoader(dataset=dataset, batch_size=4, drop_last=False)
    model = UNet().to(device)
    # model.load_state_dict(torch.load("./checkpoint/Seq2Seq/88.pth", map_location=device))

    criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if not os.path.exists(DICT_PATH):
        os.makedirs(DICT_PATH)

    counter = 0
    total_loss = 0
    total_step = 0
    model.train()
    for frame1, frame2 in dataloader:
        frame1, frame2 = frame1.to(device), frame2.to(device)

        output = model(frame1)

        loss = criterion(output, frame2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        counter += 1
        total_loss += loss.item()
        total_step += dataloader.batch_size

        average_loss = total_loss / total_step
        print(f"No.{counter} Loss: {average_loss}")

        if counter % 1000 == 0:
            torch.save(model.state_dict(), f"./checkpoint/{average_loss}.pth")


def generate_video(model, device, second, fps=1):
    model.eval()
    cap = cv2.VideoCapture(0)  # 参数0表示默认摄像头
    ret, flame = cap.read()
    cap.release()
    # 视频参数
    height, width = flame.shape[0], flame.shape[1]
    print(flame.shape)
    # 视频保存路径
    image = torch.from_numpy(np.transpose((flame / 255.0).astype(np.float32), (2, 0, 1))).to(device)
    image = torch.unsqueeze(image, dim=0)

    # 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter("./output_video.avi", fourcc, fps, (width, height))

    # 遍历图像数组并将每一帧写入视频
    for _ in range(second * fps):
        # 调整图像大小（如果需要）
        output = model(image)
        # frame = cv2.resize(frame, (width, height))
        frame = output.clone().detach().cpu().squeeze().permute(1, 2, 0).numpy()
        # 写入视频帧
        video_writer.write(frame)
        image = output.clone().detach()

    # 释放 VideoWriter 对象
    video_writer.release()


if __name__ == '__main__':
    train()
