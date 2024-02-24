#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/23 18:50
# @Author  : Jianye Liang
# @File    : train.py.py
# @Description :
import os

import torch

from model import Model
from criterion import Criterion
from dataset import TrainDataset

ROOT = os.getcwd()
DICT_PATH = "checkpoint"


def main():
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 数据集
    dataset = TrainDataset(device=device, batch_size=256)

    src_vocab_size = len(dataset.chinese.vocab)
    trg_vocab_size = len(dataset.english.vocab)

    src_pad_idx = dataset.chinese.vocab.stoi['<pad>']
    trg_pad_idx = dataset.english.vocab.stoi['<pad>']

    embedding_size = 512
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dropout = 0.1
    max_len = 120  # 最长一个句子的长度也不能超过 max_len

    forward_expansion = 2048  # pytorch官方实现的transformer中，这个参数就是线性层升维后的结果

    # 模型 # Initialize network
    model = Model(
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx, num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion, dropout, max_len, device
    ).to(device)

    # model.load_state_dict(torch.load('./checkpoint/3_20500_0.00633060569626211.pth', map_location=device))
    # 损失函数
    criterion = Criterion(ignore_index=trg_pad_idx).to(device)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 调整学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=15, verbose=True)

    if not os.path.exists(DICT_PATH):
        os.makedirs(DICT_PATH)

    total_step = 0
    total_loss = 0

    epoch = 100

    model.train()
    for i in range(1, 1 + epoch):
        print(f"No.{i} training")
        epoch_step = 0
        epoch_loss = 0
        for ch_index, en_index in dataset.iterator(train=True):
            inp_data = ch_index
            target = en_index
            # 丢弃第一个<BOS>
            output = model(inp_data, target[:-1,:])

            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            total_step += 1
            total_loss += loss
            epoch_step += 1
            epoch_loss += loss

            if total_step % 100 == 0:
                torch.save(model.state_dict(),
                           f'checkpoint/{i}_{total_step}_{total_loss / (total_step * dataset.batch_size)}.pth')

                print(f"loss:{(epoch_loss / epoch_step).item()}")

        scheduler.step(epoch_loss / epoch_step)


if __name__ == "__main__":
    main()
