#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :
import os.path
from datetime import datetime

import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader

from dataset import TrainDataset
from model import Model
from criterion import Criterion

MODEL_NAME = Model.__name__
TIME = datetime.now().strftime('%m-%d-%H-%M-%S')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(dataset=TrainDataset(), batch_size=1024, drop_last=True)
    criterion = Criterion()
    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if os.path.exists(f"./parameter/[{MODEL_NAME}][{TIME}]"):
        os.makedirs(f"./parameter/[{MODEL_NAME}][{TIME}]")

    epochs = 1000
    model.train()
    for epoch in range(1, epochs + 1):
        print(f"No.{epoch}")
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
                   f"parameter/[{MODEL_NAME}][{TIME}]/[{epoch}]-[{round(correct_num / total_num, 6)}].pth")


if __name__ == "__main__":
    main()
