#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/23 18:50
# @Author  : Jianye Liang
# @File    : module.py
# @Description :

from torch import nn


class Encoder(nn.Module):
    """
    编码器 English to Chinese
    """

    def __init__(self, encoder_embedding_num, encoder_hidden_num, en_corpus_len):
        super().__init__()
        self.embedding = nn.Embedding(en_corpus_len, encoder_embedding_num)
        # 输入数据的维度顺序为 (batch_size, seq_len, input_size)
        self.lstm = nn.LSTM(encoder_embedding_num, encoder_hidden_num, batch_first=True)

    def forward(self, en_index):
        en_embedding = self.embedding(en_index)
        _, encoder_hidden = self.lstm(en_embedding)

        return encoder_hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder_embedding_num, encoder_hidden_num, en_corpus_len, decoder_embedding_num,
                 decoder_hidden_num, ch_corpus_len):
        super(Seq2Seq, self).__init__()
        self.encoder_embedding = nn.Embedding(en_corpus_len, encoder_embedding_num)
        # 输入数据的维度顺序为 (batch_size, seq_len, input_size)
        self.encoder_lstm = nn.LSTM(encoder_embedding_num, encoder_hidden_num, batch_first=True)

        self.decoder_embedding = nn.Embedding(ch_corpus_len, decoder_embedding_num)
        self.decoder_lstm = nn.LSTM(decoder_embedding_num, encoder_hidden_num, batch_first=True)
        self.classifier = nn.Linear(decoder_hidden_num, ch_corpus_len)

    def forward(self, en_index):
        # todo
        ...
        # encoder_hidden = self.encoder(en_index)
        # decoder_output, _ = self.decoder(decoder_input, encoder_hidden)
        #
        # pre = self.classifier(decoder_output)

        return pre
