#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/17 19:17
# @Author  : Jianye Liang
# @File    : preparer.py
# @Description :
import os
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter


class Model:
    def __init__(self, model, device):
        self.correct_num = 0
        self.total_num = 0
        self.max_total_accuracy = 1e-8
        self.max_current_accuracy = 1e-8
        self.model = model.to(device)
        self.device = device
        self.hashcode = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    def load_model(self, path_to_file, only_parameter=True):
        if only_parameter:
            self.model.load_state_dict(torch.load(path_to_file, map_location=self.device))
        else:
            self.model = torch.load(path_to_file, map_location=self.device)

    def save_model(self, path='./parameter', filename='default', only_parameter=True):
        if only_parameter:
            saved_obj = self.model.state_dict()
        else:
            saved_obj = self.model
        if not os.path.exists(path):
            os.makedirs(path)
        path_to_file = os.path.join(path, filename + '.pth')
        torch.save(saved_obj, path_to_file)

    def update_condition(self, correct_num, total_num, newest=True, current_best=True, total_best=True, save=True,
                         show_info=True):
        if correct_num > total_num:
            raise ValueError()
        self.correct_num += correct_num
        self.total_num += total_num
        current_accuracy = correct_num / total_num
        if newest:
            if show_info:
                print(f'Accuracy: {current_accuracy * 100:.6f}%')
            if save:
                self.save_model(filename=f'{self.hashcode}_Newest.pth')
        if current_best and current_accuracy > self.max_current_accuracy:
            self.max_current_accuracy = current_accuracy
            if show_info:
                print(f"Current-Best -> {self.max_current_accuracy * 100:.6f}%")
            if save:
                self.save_model(filename=f'{self.hashcode}_Current_Best.pth')
        if total_best:
            accuracy = self.correct_num / self.total_num
            if accuracy > self.max_total_accuracy:
                self.max_total_accuracy = accuracy
                if show_info:
                    print(f"Total  -Best -> {self.max_total_accuracy * 100:.6f}%")
                if save:
                    self.save_model(filename=f'{self.hashcode}_Total_Best.pth')


class Writer:
    def __init__(self):
        # 数据写入器
        self.writer = None

    def create_writer(self, path='./tensorboard'):
        # tensorboard --logdir=tb/{model_name} --port=6008
        if not os.path.exists(path):
            os.makedirs(path)
        if self.writer is not None:
            self.writer.close()
        self.writer = SummaryWriter(path)

    def record_scalar(self, title, y, x):
        if self.writer is None:
            self.create_writer()
        self.writer.add_scalar((title, y, x))

    def record_images(self, images):
        ...


class Project:
    def __init__(self, device, model, criterion, dataset, dataloader, optimizer=None, scheduler=None, name=None):
        self.device = device
        self.model = model
        self.criterion = criterion
        # 数据集
        self.dataset = dataset
        # 数据加载器
        self.dataloader = dataloader
        # 学习率
        self.optimizer = optimizer
        # 余弦退火调整学习率
        self.scheduler = scheduler
        # 模型名字
        if name is None:
            self.name = self.model.__class__.__name__
        else:
            self.name = name
        # 数据写入器
        self.writer = None
