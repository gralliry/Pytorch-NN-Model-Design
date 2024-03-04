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
    dataset = TrainDataset(device=device, batch_size=2048)

    embedding_size = 32  # 512
    num_heads = 4  # 8 must be odd
    num_encoder_layers = 6  # 6
    num_decoder_layers = 6  # 6
    dropout = 0.1

    forward_expansion = 512  # 2048 pytorch官方实现的transformer中，这个参数就是线性层升维后的结果

    # 模型 # Initialize network
    model = Model(
        embedding_size=embedding_size,
        src_vocab_size=dataset.chinese_vocab_size,
        trg_vocab_size=dataset.english_vocab_size,
        src_pad_idx=dataset.CH_PAD,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        forward_expansion=forward_expansion, dropout=dropout, max_len=dataset.chinese_fix_length, device=device
    ).to(device)

    # model.load_state_dict(torch.load('./checkpoint/3_20500_0.00633060569626211.pth', map_location=device))
    # 损失函数
    criterion = Criterion(ignore_index=dataset.EN_PAD).to(device)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 调整学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=15, verbose=True)

    if not os.path.exists(DICT_PATH):
        os.makedirs(DICT_PATH)

    total_step = 0
    epoch = 100

    model.train()
    for i in range(1, 1 + epoch):
        print(f"No.{i} training")
        epoch_step = 0
        epoch_loss = 0
        for step, (inp_data, target) in enumerate(dataset.iterator(train=True), start=1):
            print(f"Step.{step}")
            # 丢弃<eos>
            output = model(inp_data, target[:, :-1])
            # [batch_size * trg_seq_length, trg_vocab_size]
            output = output.reshape(-1, output.shape[2])

            # 去除<sos>
            # [batch_size * trg_seq_length]
            target = target[:, 1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # 梯度裁剪
            optimizer.step()

            epoch_step += 1
            # 不要直接加，否则会被当成计算图的一部分
            epoch_loss += loss.item()

            total_step += 1

            if total_step % 100 == 0:

                mean_loss = epoch_loss / (epoch_step * dataset.batch_size)

                torch.save(model.state_dict(), f'checkpoint/{total_step}_{mean_loss}.pth')

                print(f"loss:{mean_loss}")

                scheduler.step(mean_loss)


if __name__ == "__main__":
    main()
