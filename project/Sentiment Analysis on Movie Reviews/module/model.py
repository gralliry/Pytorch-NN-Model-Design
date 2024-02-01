#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/1 1:41
# @Author  : Jianye Liang
# @File    : model.py
# @Description :

import torch
from torch import nn


class EmbNet(nn.Module):
    def __init__(self, emb_size, fix_length, embedding_dim=50, hidden_dim=128, output_dim=5, num_layers=2):
        super(EmbNet, self).__init__()
        self.embedding = nn.Embedding(emb_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.maxpool = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool1d(fix_length),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # 长序列
        embed = self.embedding(x)
        # embed (batch_size, embedding_dim, embedding_dim)
        x, (_, _) = self.lstm(embed)
        # x (batch_size, embedding_dim, hidden_dim * 2)
        x = torch.cat((embed, x), dim=2).permute(0, 2, 1)
        # x (batch_size, hidden_dim * 2 + embedding_dim, embedding_dim)
        # 注意力机制
        x = self.maxpool(x).squeeze(2)
        # x (batch_size, hidden_dim * 2 + embedding_dim)
        x = self.fc(x)
        # x (batch_size, output_dim)
        return x
