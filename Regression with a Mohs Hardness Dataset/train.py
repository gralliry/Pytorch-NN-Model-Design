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

HASHDIR = f"{Model.__name__}={datetime.now().strftime('%m-%d-%H-%M-%S')}"
if not os.path.exists(f"parameter/{HASHDIR}"):
    os.makedirs(f"parameter/{HASHDIR}")
CUDA = True


def main():
    device = torch.device("cuda" if torch.cuda.is_available() and CUDA else "cpu")
    dataloader = DataLoader(dataset=TrainDataset(), batch_size=1024, drop_last=True)
    criterion = Criterion()
    model = Model().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

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

            correct_num += loss.item()
            total_num += label.size(0)

        print(f"Loss: {correct_num / total_num}")
        torch.save(model.state_dict(),
                   f"parameter/{HASHDIR}/{epoch}-{round(correct_num / total_num, 6)}.pth")


if __name__ == "__main__":
    main()
