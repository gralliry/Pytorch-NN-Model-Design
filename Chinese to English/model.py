#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/23 18:50
# @Author  : Jianye Liang
# @File    : module.py
# @Description :
import torch
from torch import nn


class Model(nn.Module):
    def __init__(
            self,
            embedding_size,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
            max_len,
            device,
    ):
        super(Model, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)

        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        # (N, src_len)
        return (src == self.src_pad_idx).to(self.device)

    def forward(self, src, trg):
        batch_size, src_seq_length = src.shape
        batch_size, trg_seq_length = trg.shape

        # [seq_length, src_seq_length] range(src_seq_length) * batch_size
        src_positions = torch.arange(src_seq_length, device=self.device).repeat(batch_size, 1)

        # [seq_length, trg_seq_length] range(trg_seq_length) * batch_size
        trg_positions = torch.arange(trg_seq_length, device=self.device).repeat(batch_size, 1)

        # [batch_size, src_seq_length, embedding_size]
        embed_src = self.dropout(self.src_word_embedding(src) + self.src_position_embedding(src_positions))

        # [batch_size, trg_seq_length, embedding_size]
        embed_trg = self.dropout(self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))

        # [batch_size, src_seq_length]
        src_padding_mask = self.make_src_mask(src)

        # [trg_seq_length, trg_seq_length] left-bottom is 0, right-top is -inf
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)

        # [batch_size, trg_seq_length, embedding_size]
        out = self.transformer(embed_src, embed_trg, src_key_padding_mask=src_padding_mask, tgt_mask=trg_mask)

        # [batch_size, trg_seq_length, trg_vocab_size]
        out = self.fc_out(out)

        return out
