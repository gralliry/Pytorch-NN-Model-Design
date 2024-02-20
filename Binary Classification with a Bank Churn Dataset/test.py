#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/8 22:13
# @Author  : Gralliry
# @File    : test.py
# @Description :
import csv

import torch
from torch.utils.data.dataloader import DataLoader

from dataset import TestDataset
from model import Model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(dataset=TestDataset(), batch_size=1024)
    model = Model().to(device)
    model.load_state_dict(
        torch.load(
            "parameter/[Model]-[0209145217]-[32]-[0.8489624261856079].pth",
            map_location=device))

    pred_result = [
        ["id", "Exited"]
    ]

    model.eval()
    for data, ids in dataloader:
        data = data.to(device)
        pred = model(data)

        for sub_id, pred_index in zip(ids.tolist(), pred.argmax(dim=1).tolist()):
            pred_result.append([sub_id, pred_index])

    # 将数据写入 CSV 文件
    with open("dataset/result.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(pred_result)


if __name__ == "__main__":
    main()
