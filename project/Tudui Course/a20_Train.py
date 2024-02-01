# -*- coding: utf-8 -*-
# @Time    : 2023/11/17 21:12
# @Author  : Liang Jinaye
# @File    : 20_Train.py
# @Description :
import torch.optim
import torchvision

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from a21_Model import Tudui

DATASET_PATH = "E:\\Datasets"

# 准备数据集
train_data = torchvision.datasets.CIFAR10(DATASET_PATH, download=True, train=True,
                                          transform=torchvision.transforms.ToTensor())

test_data = torchvision.datasets.CIFAR10(DATASET_PATH, download=True, train=False,
                                         transform=torchvision.transforms.ToTensor())

# 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度为：{train_data_size}")
print(f"测试数据集的长度为：{test_data_size}")

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)
# 指定设备

# 获取可用的 CUDA 设备数量
num_devices = torch.cuda.device_count()

# 列出所有设备的名称
for i in range(num_devices):
    device_name = torch.cuda.get_device_name(i)
    print(f"Device {i}: {device_name}")
# device = torch.device("cpu")
print(f"GPU: {torch.cuda.is_available()}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:1")
# device = torch.device("cuda:2")
#
# model = Tudui().to(device)
# 网络模型，数据，损失函数
tudui = Tudui().to(device)

loss_fn = nn.CrossEntropyLoss().to(device)

learning_rate = 1e-3
optimizer = torch.optim.Adam(tudui.parameters(), lr=learning_rate)

# 设置训练网络参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练的轮数
epoch = 100

writer = SummaryWriter("./tensorboard/20")
for i in range(epoch):
    print(f"第 {i + 1} 轮训练开始")
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = tudui(imgs).to(device)
        loss = loss_fn(output, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1

        if total_train_step % 1000 == 0:
            print(f"训练次数: {total_train_step}, Loss: {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试
    tudui.eval()
    total_test_loss = 0
    # 整体正确率
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            targets = targets.cuda()
            output = tudui(imgs.cuda())
            loss = loss_fn(output.cuda(), targets)
            total_test_loss += loss.item()
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print(f"整体测试集上的loss: {total_test_loss / test_data_size}")
    writer.add_scalar("test_loss", total_test_loss / test_data_size, total_test_step)
    print(f"整体测试集上的正确率: {total_accuracy / test_data_size}")
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    torch.save(tudui, f"./parameter/a20_train/{i+1}.pth")
# tensorboard --logdir=tb/13 --port=6008
writer.close()
