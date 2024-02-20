#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/8 19:26
# @Author  : Gralliry
# @File    : dataset.py
# @Description :
import csv
import torch

from torch.utils.data import Dataset

gender = {"Male": 0, "Female": 1}
family_history_with_overweight = {"no": 0, "yes": 1}
FAVC = {"no": 0, "yes": 1}
CAEC = {"no": 0, "Sometimes": 1, "Always": 2, "Frequently": 3}
SMOKE = {"no": 0, "yes": 1}
SCC = {"no": 0, "yes": 1}
CALC = {"no": 0, "Sometimes": 1, "Always": 2, "Frequently": 2}
MTRANS = {"Automobile": 0, "Bike": 1, "Motorbike": 2, "Public_Transportation": 3, "Walking": 4}
NObeyesdad = {"Insufficient_Weight": 0, "Normal_Weight": 1, "Obesity_Type_I": 2, "Obesity_Type_II": 3,
              "Obesity_Type_III": 4, "Overweight_Level_I": 5, "Overweight_Level_II": 6}

NObeyesdad_Re = ["Insufficient_Weight", "Normal_Weight", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III",
                 "Overweight_Level_I", "Overweight_Level_II"]


class TrainDataset(Dataset):
    def __init__(self):
        self.data = []
        self.label = []
        with open("dataset/train.csv", 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                self.data.append([
                    gender[row[1]],
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    family_history_with_overweight[row[5]],
                    FAVC[row[6]],
                    float(row[7]),
                    float(row[8]),
                    CAEC[row[9]],
                    SMOKE[row[10]],
                    float(row[11]),
                    SCC[row[12]],
                    float(row[13]),
                    float(row[14]),
                    CALC[row[15]],
                    MTRANS[row[16]]
                ])
                self.label.append(NObeyesdad[row[17]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], torch.tensor(self.label[item])


class TestDataset(Dataset):

    def __init__(self):
        self.data = []
        self.id = []
        with open("dataset/test.csv", 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                self.data.append([
                    gender[row[1]],
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    family_history_with_overweight[row[5]],
                    FAVC[row[6]],
                    float(row[7]),
                    float(row[8]),
                    CAEC[row[9]],
                    SMOKE[row[10]],
                    float(row[11]),
                    SCC[row[12]],
                    float(row[13]),
                    float(row[14]),
                    CALC[row[15]],
                    MTRANS[row[16]]
                ])
                self.id.append(int(row[0]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.id[item]
