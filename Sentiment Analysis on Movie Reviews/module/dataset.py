#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/1 1:45
# @Author  : Jianye Liang
# @File    : dataset.py
# @Description :
import os

from torchtext.data import Field, TabularDataset, BucketIterator

DATASET_PATH = 'E:\\Datasets\\sentiment-analysis-on-movie-reviews'


# torchtext已经继承dataset，这里不用重复继承
class MyDataset:
    def __init__(self, device, batch_size=1, split_ratio=0.9):
        # 定义文本字段 # 设置 include_lengths=True
        self.text = Field(sequential=True, tokenize=lambda x: x.split(), lower=True, batch_first=True, fix_length=30)
        self.label = Field(sequential=False, tokenize=lambda x: int(x) - 1, use_vocab=False, batch_first=True)
        self.PhraseId = Field(sequential=False, use_vocab=False)

        # 加载数据集
        train_data_set = TabularDataset(
            path=os.path.join(DATASET_PATH, 'train.tsv'),
            format='tsv',
            fields=[('PhraseId', None), ('SentenceId', None), ('Phrase', self.text), ('Sentiment', self.label)],
            skip_header=True,  # 是否跳过 TSV 文件的头部
        )

        train_data, valid_data = train_data_set.split(split_ratio=split_ratio)

        test_data = TabularDataset(
            path=os.path.join(DATASET_PATH, 'test.tsv'),
            format='tsv',
            fields=[('PhraseId', self.PhraseId), ('SentenceId', None), ('Phrase', self.text), ('Sentiment', None)],
            skip_header=True
        )

        # 构建词汇表
        self.text.build_vocab(train_data)

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

        self.test_iterator = BucketIterator.splits(
            (test_data,),
            batch_size=batch_size,
            shuffle=False,
            device=device
        )[0]

    def get_train_iterator(self, train=True):
        data_iterator = self.train_iterator if train else self.valid_iterator
        for batch in data_iterator:
            yield batch.Phrase, batch.Sentiment

    def get_test_iterator(self):
        for batch in self.test_iterator:
            yield batch.PhraseId, batch.Phrase
