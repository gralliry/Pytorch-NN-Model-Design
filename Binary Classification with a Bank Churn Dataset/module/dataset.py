#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/8 19:26
# @Author  : Gralliry
# @File    : dataset.py
# @Description :
import os
import csv
import torch

from torch.utils.data import Dataset

DATASET_PATH = "E:\\Datasets\\Binary Classification with a Bank Churn Dataset"

Geography = {"France": 0, "Spain": 1, "Germany": 2}
Gender = {"Male": 0, "Female": 1}


class TrainDataset(Dataset):
    def __init__(self):
        self.data = []
        self.label = []
        with open(os.path.join(DATASET_PATH, "train.csv"), 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                self.data.append([
                    float(row[3]),
                    Geography[row[4]],
                    Gender[row[5]],
                    float(row[6]),
                    float(row[7]),
                    float(row[8]),
                    float(row[9]),
                    float(row[10]),
                    float(row[11]),
                    float(row[12]),
                ])
                self.label.append(int(row[13]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # print(self.data[item], self.label[item])
        return torch.tensor(self.data[item]), torch.tensor(self.label[item])


class TestDataset(Dataset):

    def __init__(self):
        self.data = []
        self.id = []
        with open(os.path.join(DATASET_PATH, "test.csv"), 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                self.data.append([
                    float(row[3])/1000,
                    Geography[row[4]]/3,
                    Gender[row[5]],
                    float(row[6])/40,
                    float(row[7])/10,
                    float(row[8])/100000,
                    float(row[9])/4,
                    float(row[10]),
                    float(row[11]),
                    float(row[12])/100000,
                ])
                self.id.append(int(row[0]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return torch.tensor(self.data[item]), torch.tensor(self.id[item])
