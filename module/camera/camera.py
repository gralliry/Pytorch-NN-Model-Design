#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :
import cv2
import numpy as np


class Camera:
    def __init__(self, device=0, size=(640, 480)):
        # size (width, height)
        self.cap = cv2.VideoCapture(device)  # 参数0表示默认摄像头
        self.size = size
        self.frame = None

    def get(self, origin=False):
        # 从摄像头读取图像
        ret, frame = self.cap.read()
        # 调整大小为新的宽度和高度
        frame = cv2.resize(frame, self.size)
        self.frame = frame
        return frame
        # frame = np.transpose(frame, (2, 0, 1))
        # return (frame / 255.0).astype(np.float32)

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def __del__(self):
        self.close()

    def show(self, reget=False):
        if reget:
            self.get(origin=True)
        # 显示图像
        cv2.imshow("Webcam", self.frame)
        # 按下 'q' 键退出循环
        cv2.waitKey(10)


if __name__ == "__main__":
    camera = Camera(device=0)
    while True:
        camera.show(reget=True)
