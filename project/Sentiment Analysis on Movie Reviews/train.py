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

from module import TextRCNN, TextRNN_Att, MyDataset

CUDA = True

DICT_PATH = "./parameter"


def main():
    device = torch.device('cuda' if torch.cuda.is_available() and CUDA else 'cpu')
    print("Using", device)

    dataset = MyDataset(device=device, batch_size=2048)
    # 实例化模型
    run_model = TextRCNN(len(dataset.text.vocab)).to(device)

    HASHCODE = f"{run_model.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    # 使用预训练的词嵌入初始化嵌入层
    # pretrained_embeddings = TEXT.vocab.vectors
    # run_model.embedding.weight.data.copy_(pretrained_embeddings)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(run_model.parameters(), lr=0.001)

    if not os.path.exists(DICT_PATH):
        os.makedirs(DICT_PATH)

    # 评估模型
    correct_num = 0
    total_num = 0
    max_total_accuracy = 1e-8
    max_current_accuracy = 1e-8

    # 训练模型
    num_epochs = 1000
    for epoch in range(1, num_epochs + 1):
        print(f"No.{epoch} training...")
        run_model.train()
        for text, labels in dataset.get_train_iterator(train=True):
            predictions = run_model(text)

            loss = criterion(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        current_correct_num = 0
        current_total_num = 0

        run_model.eval()
        with torch.no_grad():
            for text, labels in dataset.get_train_iterator(train=False):
                predictions = run_model(text)

                current_correct_num += (predictions.argmax(1) == labels).sum().item()
                current_total_num += labels.size(0)

        current_accuracy = current_correct_num / current_total_num
        print(f'Test Accuracy: {current_accuracy * 100:.6f}%')
        torch.save(run_model.state_dict(), f"./parameter/{HASHCODE}_newest.pth")
        # 保存当前批次最好的训练结果
        if current_accuracy > max_current_accuracy:
            max_current_accuracy = current_accuracy
            print(f"Current-Best -> {max_current_accuracy * 100:.6f}%")
            torch.save(run_model.state_dict(), f"./parameter/{HASHCODE}_current_best.pth")
        correct_num += current_correct_num
        total_num += current_total_num
        accuracy = correct_num / total_num
        # 保存最好的训练结果
        if accuracy > max_total_accuracy:
            max_total_accuracy = accuracy
            print(f"Total  -Best -> {max_total_accuracy * 100:.6f}%")
            torch.save(run_model.state_dict(), f"./parameter/{HASHCODE}_total_best.pth")


if __name__ == "__main__":
    main()
