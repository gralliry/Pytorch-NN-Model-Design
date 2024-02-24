#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/23 18:50
# @Author  : Jianye Liang
# @File    : train.py.py
# @Description :
import os

import torch
import torch.nn as nn

from model import Seq2Seq
from dataset import MyDataset

ROOT = os.getcwd()
DICT_PATH = "checkpoint"


def translate(sentence, model, dataset: MyDataset):
    # batch_size = 1

    # 翻译成英语索引
    en_index = dataset.english_2_vector(sentence).unsqueeze(0)

    chinese_vector = model(en_index)
    chinese_words = dataset.vector_2_chinese(chinese_vector, join=True)

    return chinese_words


def main():
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 数据集
    dataset = MyDataset(device=device, batch_size=256)

    # 模型
    model = Seq2Seq(
        encoder_embedding_num=100, encoder_hidden_num=100, en_corpus_len=len(dataset.english.vocab),
        decoder_embedding_num=100, decoder_hidden_num=100, ch_corpus_len=len(dataset.chinese.vocab)
    ).to(device)

    # model.load_state_dict(torch.load('./checkpoint/3_20500_0.00633060569626211.pth', map_location=device))
    # 损失函数
    criterion = nn.CrossEntropyLoss().to(device)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 余弦退火调整学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    if not os.path.exists(DICT_PATH):
        os.makedirs(DICT_PATH)

    total_step = 0
    total_loss = 0

    epoch = 100
    for i in range(1, 1 + epoch):
        print(f"No.{i} training")
        for en_index, ch_index in dataset.iterator(train=True):
            # 丢弃第一个<BOS>
            output = model(en_index, ch_index)

            loss = criterion(output, ch_index)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_step += 1
            total_loss += loss.item()

            if total_step % 100 == 0:
                torch.save(model.state_dict(),
                           f'checkpoint/{i}_{total_step}_{total_loss / (total_step * dataset.batch_size)}.pth')
                scheduler.step()
                print(f"loss:{total_loss / (total_step * dataset.batch_size)}")

    while True:
        s = input("请输入英文: ")
        print("译文:", translate(s, model, dataset))


if __name__ == "__main__":
    main()
