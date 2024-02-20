#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/18 11:46
# @Author  : Jianye Liang
# @File    : dataset.py
# @Description :

from collections import deque

import cv2
import numpy as np
from torch.utils.data import Dataset


class CvVideoDataset(Dataset):
    def __init__(self, sequence_length=10, size=(640, 480)):
        self.cap = cv2.VideoCapture(0)  # 参数0表示默认摄像头
        self.size = size[::-1]
        self.sequence_length = sequence_length
        self.video = deque()
        for _ in range(len(self.video)-self.sequence_length):
            self.video.append(self.get_flash())

    def __len__(self):
        # return 1
        return 999999999999

    def __getitem__(self, index):
        video1 = np.array(list(self.video))
        self.get_flash()
        video2 = np.array(list(self.video))
        return video1, video2

    def get_flash(self):
        # 从摄像头读取图像
        ret, frame = self.cap.read()
        # 显示图像
        cv2.imshow("Webcam", frame)
        # 按下 'q' 键退出循环
        cv2.waitKey(1)
        frame = np.reshape(frame, (*self.size, 3))
        frame = np.transpose(frame, (2, 0, 1))
        self.video.append(frame)
        self.video.popleft()
        return frame

    def __del__(self):
        self.cap.release()


class CvImageDataset(Dataset):
    def __init__(self, size=(640, 480), transforms=None, is_show=False):
        self.cap = cv2.VideoCapture(0)  # 参数0表示默认摄像头
        self.size = size[::-1]
        self.transforms = transforms
        self.is_show = is_show

    def __len__(self):
        # return 1
        return 999999999999

    def __getitem__(self, index):
        frame1 = np.array(self.get_flash())
        frame2 = np.array(self.get_flash())
        return frame1, frame2

    def get_flash(self):
        # 从摄像头读取图像
        ret, frame = self.cap.read()

        if self.is_show:
            # 显示图像
            cv2.imshow("Webcam", frame)
            # 按下 'q' 键退出循环
            cv2.waitKey(1)
        frame = np.reshape(frame, (*self.size, 3))
        frame = np.transpose(frame, (2, 0, 1))
        return (frame / 255.0).astype(np.float32)

    def __del__(self):
        self.cap.release()


if __name__ == '__main__':
    datasets = CvImageDataset()

    for image, target in datasets:
        print(image.shape, type(image))

    # 定义数据转换
    # data_transform =transform=transforms.Compose([
    #     transforms.Resize((480, 480)),
    #     transforms.ToTensor(),
    # ])
