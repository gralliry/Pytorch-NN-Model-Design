#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/28 23:45
# @Author  : Jianye Liang
# @File    : train.py
# @Description :
import os.path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import MyDataset
from model import TextCNN

CUDA = True

DICT_PATH = "checkpoint"


def main():
    device = torch.device('cuda' if torch.cuda.is_available() and CUDA else 'cpu')
    print("Using", device)

    dataset = MyDataset(device=device, batch_size=2048)
    # 实例化模型
    model = TextCNN(
        len(dataset.text.vocab),
        padding_index=dataset.text.vocab.stoi['<pad>']
    ).to(device)

    HASHCODE = f"{model.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    # 使用预训练的词嵌入初始化嵌入层
    # pretrained_embeddings = TEXT.vocab.vectors
    # model.embedding.weight.data.copy_(pretrained_embeddings)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if not os.path.exists(DICT_PATH):
        os.makedirs(DICT_PATH)

    # 训练模型
    num_epochs = 1000
    for epoch in range(1, num_epochs + 1):
        print(f"No.{epoch} training...")
        model.train()
        for text, labels in dataset.get_train_iterator(train=True):
            predictions = model(text)

            loss = criterion(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 评估模型
        correct_num = 0
        total_num = 0
        model.eval()
        with torch.no_grad():
            for text, labels in dataset.get_train_iterator(train=False):
                predictions = model(text)

                correct_num += (predictions.argmax(1) == labels).sum().item()
                total_num += labels.size(0)
        print(f'Test Accuracy: {correct_num / total_num * 100:.6f}%')
        torch.save(model.state_dict(), f"checkpoint/{HASHCODE}_{epoch}_{correct_num / total_num}.pth")


if __name__ == "__main__":
    main()
