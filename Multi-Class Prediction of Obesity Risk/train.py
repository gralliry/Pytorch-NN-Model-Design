#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/8 20:18
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(dataset=TrainDataset(), batch_size=1024, drop_last=True)
    criterion = nn.CrossEntropyLoss()
    model = Model().to(device)
    HASHCODE = f"{model.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 1000
    model.train()
    for epoch in range(1, epochs + 1):
        print(epoch)
        correct_num = 0
        total_num = 0
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            pred = model(data)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct_num += (pred.argmax(dim=1) == label).sum()
            total_num += label.size(0)

        print(f"Accuracy: {correct_num / total_num}")
        torch.save(model.state_dict(),
                   f"parameter/{HASHCODE}_{epoch}_{correct_num / total_num}.pth")


if __name__ == "__main__":
    main()
