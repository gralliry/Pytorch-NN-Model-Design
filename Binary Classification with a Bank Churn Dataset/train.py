#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/8 20:19
# @Author  : Gralliry
# @File    : train.py
# @Description :
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader

from dataset import TrainDataset
from model import Model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() and True else "cpu")
    dataloader = DataLoader(dataset=TrainDataset(), batch_size=2048, drop_last=True)
    criterion = nn.BCELoss()
    model = Model().to(device)
    HASHCODE = f"[{model.__class__.__name__}]-[{datetime.now().strftime('%m%d%H%M%S')}]"
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    epochs = 1000
    for epoch in range(1, epochs + 1):
        print(f"No.{epoch}")
        correct_num = 0
        total_num = 0

        for data, label in dataloader:
            data, label = data.to(device), label.to(device)

            # Z-score 标准化
            mean = data.mean(dim=0)
            std = data.std(dim=0)
            data = (data - mean) / std

            pred = model(data)


            # print(pred.shape, label.shape)
            # print(pred)
            # print(label)
            # input("_"*50)
            loss = criterion(torch.round(pred), label.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct_num += (torch.round(pred) == label).sum()
            total_num += label.size(0)

        print(f"Accuracy: {correct_num / total_num}")
        torch.save(model.state_dict(),
                   f"parameter/{HASHCODE}-[{epoch}]-[{correct_num / total_num}].pth")


if __name__ == "__main__":
    main()
