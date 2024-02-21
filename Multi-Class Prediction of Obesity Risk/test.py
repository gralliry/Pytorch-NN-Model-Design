#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/8 22:13
# @Author  : Gralliry
# @File    : test.py
# @Description :
import csv

import torch
from torch.utils.data.dataloader import DataLoader
from module import TestDataset, Model, NObeyesdad_Re


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(dataset=TestDataset(), batch_size=1024)
    model = Model().to(device)
    model.load_state_dict(
        torch.load(
            "parameter/Model_2024-02-08-22-33-38_160_0.895068347454071.pth",
            map_location=device))

    pred_result = [
        ["id", "NObeyesdad"]
    ]

    model.eval()
    for data, id in dataloader:
        data = data.to(device)
        pred = model(data)

        for id, pred_index in zip(id.tolist(), pred.argmax(dim=1).tolist()):
            pred_result.append([id, NObeyesdad_Re[pred_index]])

    # 将数据写入 CSV 文件
    with open("./result.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(pred_result)


if __name__ == "__main__":
    main()
