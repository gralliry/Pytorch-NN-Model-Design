#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :
import csv

import torch
from torch.utils.data.dataset import Dataset


class TrainDataset(Dataset):
    def __init__(self):
        self.data = []
        self.label = []
        with open("dataset/train.csv", 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                self.data.append([
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                    float(row[6]),
                    float(row[7]),
                    float(row[8]),
                    float(row[9]),
                    float(row[10]),
                    float(row[11]),
                ])
                self.label.append(float(row[12]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return torch.tensor(self.data[item]), torch.tensor(self.label[item])


class TestDataset(Dataset):

    def __init__(self):
        self.data = []
        self.id = []
        with open("dataset/test.csv", 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                self.data.append([
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                    float(row[6]),
                    float(row[7]),
                    float(row[8]),
                    float(row[9]),
                    float(row[10]),
                    float(row[11]),
                ])
                self.id.append(int(row[0]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return torch.tensor(self.data[item]), torch.tensor(self.id[item])
