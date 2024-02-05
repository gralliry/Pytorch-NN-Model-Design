#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/1 1:41
# @Author  : Jianye Liang
# @File    : model.py
# @Description :

import torch
from torch import nn
import torch.nn.functional as f
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextRCNN(nn.Module):
    def __init__(self, emb_size, embedding_dim=100, hidden_dim=256, output_dim=5, num_layers=3, fix_length=30):
        super(TextRCNN, self).__init__()
        self.embedding = nn.Embedding(emb_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.maxpool = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool1d(fix_length)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + embedding_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim),
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
        # print(x.shape)
        x = self.maxpool(x).squeeze(2)
        # x (batch_size, hidden_dim * 2 + embedding_dim)
        x = self.fc(x)
        # x (batch_size, output_dim)
        return x


class TextRNN_Att(nn.Module):
    def __init__(self, emb_size, embedding_dim=50, hidden_dim=128, output_dim=5, num_layers=2):
        super(TextRNN_Att, self).__init__()
        self.embedding = nn.Embedding(emb_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            bidirectional=True, batch_first=True)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(hidden_dim * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # print(x.shape)
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = f.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [1024, 50, 256]
        out = torch.sum(out, 1)  # [1024, ?, 256]
        out = f.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [1024, 5]
        return out


class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=50, num_output=5, rnn_model='LSTM', use_last=True, embedding_tensor=None,
                 hidden_size=128, num_layers=1):
        super(TextRNN, self).__init__()
        self.use_last = use_last
        # embedding
        self.encoder = None
        if torch.is_tensor(embedding_tensor):
            self.encoder = nn.Embedding(vocab_size, embed_size, _weight=embedding_tensor)
            self.encoder.weight.requires_grad = False
        else:
            self.encoder = nn.Embedding(vocab_size, embed_size)

        self.drop_en = nn.Dropout(p=0.6)

        # rnn module
        if rnn_model == 'LSTM':
            self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.5,
                               batch_first=True, bidirectional=True)
        elif rnn_model == 'GRU':
            self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.5,
                              batch_first=True, bidirectional=True)
        else:
            raise LookupError(' only support LSTM and GRU')

        self.bn2 = nn.BatchNorm1d(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, num_output)

    def forward(self, x, seq_lengths):
        '''
        Args:
            x: (batch, time_step, input_size)

        Returns:
            num_output size
        '''

        x_embed = self.encoder(x)
        x_embed = self.drop_en(x_embed)
        packed_input = pack_padded_sequence(x_embed, seq_lengths.cpu().numpy(), batch_first=True)

        # r_out shape (batch, time_step, output_size)
        # None is for initial hidden state
        packed_output, ht = self.rnn(packed_input, None)
        out_rnn, _ = pad_packed_sequence(packed_output, batch_first=True)

        row_indices = torch.arange(0, x.size(0)).long()
        col_indices = seq_lengths - 1
        if next(self.parameters()).is_cuda:
            row_indices = row_indices.cuda()
            col_indices = col_indices.cuda()

        if self.use_last:
            last_tensor = out_rnn[row_indices, col_indices, :]
        else:
            # use mean
            last_tensor = out_rnn[row_indices, :, :]
            last_tensor = torch.mean(last_tensor, dim=1)

        fc_input = self.bn2(last_tensor)
        out = self.fc(fc_input)
        return out
