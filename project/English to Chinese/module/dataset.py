#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/23 18:51
# @Author  : Jianye Liang
# @File    : dataset.py
# @Description :
import os
import logging

from torchtext.data import Field, TabularDataset, BucketIterator
import jieba

jieba.setLogLevel(logging.INFO)

DATASET_PATH = 'E:\\Datasets\\translation2019zh'


# torchtext已经继承dataset，这里不用重复继承
class MyDataset:

    def __init__(self, device, batch_size: int = 1, split_ratio: float = 0.8):
        self.device = device
        self.batch_size = batch_size
        # 定义文本字段 # 设置 include_lengths=True
        self.english = Field(sequential=True, tokenize=lambda x: x.split(), lower=True, batch_first=True, fix_length=50)
        self.chinese = Field(sequential=True, tokenize=lambda x: list(jieba.cut(x)), batch_first=True, fix_length=50)

        # 加载数据集
        train_data_set = TabularDataset(
            path=os.path.join(DATASET_PATH, 'sentences_train.tsv'),
            format='tsv',
            fields=[('english', self.english), ('chinese', self.chinese)],
            skip_header=True,  # 是否跳过 TSV 文件的头部
        )

        train_data, valid_data = train_data_set.split(split_ratio=split_ratio)

        # 构建词汇表
        self.english.build_vocab(train_data)
        self.chinese.build_vocab(train_data)

        # 创建迭代器
        self.train_iterator, self.valid_iterator = BucketIterator.splits(
            (train_data, valid_data),
            batch_size=batch_size,
            sort_key=lambda x: len(x.Phrase),
            sort_within_batch=False,
            shuffle=True,
            repeat=False,
            device=device,
        )

    def iterator(self, train=True):
        data_iterator = self.train_iterator if train else self.valid_iterator
        for batch in data_iterator:
            yield batch.english, batch.chinese

    def vector_2_english(self, vector, join=False):
        words = []
        for idx in vector:
            # 使用vocab.itos将索引转为词语
            words.append(self.english.vocab.itos[idx.item()])
        return " ".join(words) if join else words

    def vector_2_chinese(self, vector, join=False):
        words = []
        for idx in vector:
            # 使用vocab.itos将索引转为词语
            words.append(self.chinese.vocab.itos[idx.item()])
        return "".join(words) if join else words

    def english_2_vector(self, sentence):
        # 使用 TEXT 对象进行处理
        processed_sentence = self.english.preprocess(sentence)  # 分词、转小写等
        # 将处理后的文本转为索引序列
        indexed_sentence = self.english.process([processed_sentence])
        # 将索引序列转为 PyTorch Tensor
        input_tensor = self.english.numericalize([indexed_sentence]).to(self.device)
        return input_tensor

    def chinese_2_vector(self, sentence):
        # 使用 TEXT 对象进行处理
        processed_sentence = self.chinese.preprocess(sentence)  # 分词、转小写等
        # 将处理后的文本转为索引序列
        indexed_sentence = self.chinese.process([processed_sentence])
        # 将索引序列转为 PyTorch Tensor
        input_tensor = self.chinese.numericalize([indexed_sentence]).to(self.device)
        return input_tensor
