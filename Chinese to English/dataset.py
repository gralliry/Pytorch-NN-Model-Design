#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/23 18:51
# @Author  : Jianye Liang
# @File    : dataset.py
# @Description :
import json
import os

import torch
from torchtext.data import Field, TabularDataset, BucketIterator

from utils import chinese_tokenizer, english_tokenizer, chinese_detokenizer, english_detokenizer

DATASET_PATH = 'E:\\Datasets\\translation2019zh/'


# torchtext已经继承dataset，这里不用重复继承
class TrainDataset:

    def __init__(self, device, batch_size: int = 1, english_fix_length=50, chinese_fix_length=50):
        self.device = device
        self.batch_size = batch_size
        # 定义文本字段 # 设置 include_lengths=True
        self.english = Field(sequential=True, tokenize=english_tokenizer, fix_length=english_fix_length, lower=True,
                             init_token='<sos>', eos_token='<eos>')
        self.chinese = Field(sequential=True, tokenize=chinese_tokenizer, fix_length=chinese_fix_length, lower=True,
                             init_token='<sos>', eos_token='<eos>')

        # 加载数据集
        train_data, valid_data = TabularDataset(
            path=DATASET_PATH,
            train='sentences_train.tsv',
            test="sentences_valid.tsv",
            format='tsv',
            fields=[('english', self.english), ('chinese', self.chinese)],
            skip_header=True,  # 是否跳过 TSV 文件的头部
        )

        # 构建词汇表
        self.english.build_vocab(train_data, max_size=30000, min_freq=2)
        self.chinese.build_vocab(train_data, max_size=30000, min_freq=2)

        with open(os.path.join(DATASET_PATH, "english_stoi.json"), "w", encoding="utf-8") as file:
            json.dump(dict(self.english.vocab.stoi), file, indent=4)
        with open(os.path.join(DATASET_PATH, "english_itos.json"), "w", encoding="utf-8") as file:
            json.dump(list(self.english.vocab.itos), file, indent=4)
        with open(os.path.join(DATASET_PATH, "chinese_stoi.json"), "w", encoding="utf-8") as file:
            json.dump(dict(self.chinese.vocab.stoi), file, indent=4)
        with open(os.path.join(DATASET_PATH, "chinese_itos.json"), "w", encoding="utf-8") as file:
            json.dump(list(self.chinese.vocab.itos), file, indent=4)

        self.EN_SOS = self.english.vocab.stoi['<sos>']
        self.EN_EOS = self.english.vocab.stoi['<eos>']
        self.EN_PAD = self.english.vocab.stoi['<pad>']

        self.CH_SOS = self.chinese.vocab.stoi['<sos>']
        self.CH_EOS = self.chinese.vocab.stoi['<eos>']
        self.CH_PAD = self.chinese.vocab.stoi['<pad>']

        # 创建迭代器
        self.train_dataloader, self.valid_dataloader = BucketIterator.splits(
            (train_data, valid_data),
            batch_size=batch_size,
            sort_key=lambda x: len(x.chinese),
            sort_within_batch=True,
            device=device,
        )

    def iterator(self, train=True):
        for batch in (self.train_dataloader if train else self.valid_dataloader):
            yield batch.chinese, batch.english

    def vector_2_english(self, vector, join=False):
        if isinstance(vector, torch.Tensor):
            vector = vector.tolist()
        # 使用vocab.itos将索引转为词语
        words = [self.english.vocab.itos[idx] for idx in vector]
        return english_detokenizer(words) if join else words

    def vector_2_chinese(self, vector, join=False):
        if isinstance(vector, torch.Tensor):
            vector = vector.tolist()
        # 使用vocab.itos将索引转为词语
        words = [self.chinese.vocab.itos[idx] for idx in vector]
        return chinese_detokenizer(words) if join else words

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
        print(processed_sentence)
        print(processed_sentence.shape)
        # 将处理后的文本转为索引序列
        indexed_sentence = self.chinese.process([processed_sentence])
        print(indexed_sentence)
        print(indexed_sentence.shape)
        # 将索引序列转为 PyTorch Tensor
        input_tensor = self.chinese.numericalize([indexed_sentence]).to(self.device)
        print(input_tensor)
        print(input_tensor.shape)
        return input_tensor


class TestDataset:
    def __init__(self):
        with open(os.path.join(DATASET_PATH, "english_stoi.json"), "r", encoding="utf-8") as file:
            self.english_stoi = json.load(file)
        with open(os.path.join(DATASET_PATH, "english_itos.json"), "r", encoding="utf-8") as file:
            self.english_itos = json.load(file)
        with open(os.path.join(DATASET_PATH, "chinese_stoi.json"), "r", encoding="utf-8") as file:
            self.chinese_stoi = json.load(file)
        with open(os.path.join(DATASET_PATH, "chinese_itos.json"), "r", encoding="utf-8") as file:
            self.chinese_itos = json.load(file)

        self.EN_SOS = self.english_stoi['<sos>']
        self.EN_EOS = self.english_stoi['<eos>']
        self.EN_PAD = self.english_stoi['<pad>']

        self.CH_SOS = self.chinese_stoi['<sos>']
        self.CH_EOS = self.chinese_stoi['<eos>']
        self.CH_PAD = self.chinese_stoi['<pad>']

        self.english_vocab_size = len(self.english_stoi)
        self.chinese_vocab_size = len(self.chinese_stoi)
