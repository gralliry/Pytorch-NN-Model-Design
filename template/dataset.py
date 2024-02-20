#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :

import csv
import torch

from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self):
        self.data = []
        self.label = []

        with open("../dataset/train.csv", 'r') as file:
            reader = csv.reader(file)

        next(reader)
        for row in reader:
            self.data.append([
                ...
            ])
            self.label.append(...)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return torch.tensor(self.data[item]), torch.tensor(self.label[item])


class TestDataset(Dataset):

    def __init__(self):
        self.data = []
        self.id = []

        with open("../dataset/test.csv", 'r') as file:
            reader = csv.reader(file)

        next(reader)
        for row in reader:
            self.data.append([
                ...
            ])
            self.id.append(...)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return torch.tensor(self.data[item]), torch.tensor(self.id[item])
