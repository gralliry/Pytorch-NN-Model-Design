#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :

import csv

import torch
from torch.utils.data.dataloader import DataLoader
from module import TestDataset, Model

HASHDIR = "Model=02-20-00-41-54"
FILENAME = "100-0.00189.pth"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(dataset=TestDataset(), batch_size=1024)
    model = Model().to(device)
    model.load_state_dict(
        torch.load(f"./parameter/{HASHDIR}/{FILENAME}", map_location=device))
    pred_result = [
        ["id", "Hardness"]
    ]

    model.eval()
    for data, id in dataloader:
        data = data.to(device)
        pred = model(data)

        for id, result in zip(id.tolist(), pred.tolist()):
            pred_result.append([id, round(result, 3)])

    # 将数据写入 CSV 文件
    with open("./result.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(pred_result)


if __name__ == "__main__":
    main()
