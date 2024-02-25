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

DATASET_PATH = "E:\\Datasets\\translation2019zh"


# torchtext已经继承dataset，这里不用重复继承
class TrainDataset:

    def __init__(self, device, batch_size: int = 1, english_fix_length=30, chinese_fix_length=30):
        self.device = device
        self.batch_size = batch_size
        self.chinese_fix_length = chinese_fix_length
        self.english_fix_length = english_fix_length
        # 定义文本字段 # 设置 include_lengths=True
        english = Field(sequential=True, tokenize=english_tokenizer, fix_length=english_fix_length, lower=True,
                        init_token='<sos>', eos_token='<eos>', batch_first=True)
        chinese = Field(sequential=True, tokenize=chinese_tokenizer, fix_length=chinese_fix_length, lower=True,
                        init_token='<sos>', eos_token='<eos>', batch_first=True)

        # 加载数据集
        # PermissionError: [Errno 13] Permission denied ? must use ".splits(...)"
        train_data, valid_data = TabularDataset.splits(
            path=DATASET_PATH,
            train='sentences_train.tsv',
            test="sentences_valid.tsv",
            format='tsv',
            fields=[('english', english), ('chinese', chinese)],
            skip_header=True,  # 是否跳过 TSV 文件的头部
        )

        # 构建词汇表
        english.build_vocab(train_data, max_size=10000, min_freq=5)
        chinese.build_vocab(train_data, max_size=10000, min_freq=5)

        self.english_vocab_size = len(english.vocab)
        self.chinese_vocab_size = len(chinese.vocab)

        with open("./dataset/english_stoi.json", "w", encoding="utf-8") as file:
            json.dump(dict(english.vocab.stoi), file, indent=4, ensure_ascii=False)
        with open("./dataset/english_itos.json", "w", encoding="utf-8") as file:
            json.dump(list(english.vocab.itos), file, indent=4, ensure_ascii=False)
        with open("./dataset/chinese_stoi.json", "w", encoding="utf-8") as file:
            json.dump(dict(chinese.vocab.stoi), file, indent=4, ensure_ascii=False)
        with open("./dataset/chinese_itos.json", "w", encoding="utf-8") as file:
            json.dump(list(chinese.vocab.itos), file, indent=4, ensure_ascii=False)

        self.EN_SOS = english.vocab.stoi['<sos>']
        self.EN_EOS = english.vocab.stoi['<eos>']
        self.EN_PAD = english.vocab.stoi['<pad>']

        self.CH_SOS = chinese.vocab.stoi['<sos>']
        self.CH_EOS = chinese.vocab.stoi['<eos>']
        self.CH_PAD = chinese.vocab.stoi['<pad>']

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
            # [batch_size, seq_length], [batch_size, seq_length]
            yield batch.chinese, batch.english


class TestDataset:
    def __init__(self, device, chinese_fix_length=50, english_fix_length=50):
        self.device = device
        self.chinese_fix_length = chinese_fix_length
        self.english_fix_length = english_fix_length
        with open(os.path.join(DATASET_PATH, "english_stoi.json"), "r", encoding="utf-8") as file:
            self.english_stoi = json.load(file)
        with open(os.path.join(DATASET_PATH, "english_itos.json"), "r", encoding="utf-8") as file:
            self.english_itos = json.load(file)
        with open(os.path.join(DATASET_PATH, "chinese_stoi.json"), "r", encoding="utf-8") as file:
            self.chinese_stoi = json.load(file)
        with open(os.path.join(DATASET_PATH, "chinese_itos.json"), "r", encoding="utf-8") as file:
            self.chinese_itos = json.load(file)

        self.EN_UNK = self.english_stoi['<unk>']
        self.EN_SOS = self.english_stoi['<sos>']
        self.EN_EOS = self.english_stoi['<eos>']
        self.EN_PAD = self.english_stoi['<pad>']

        self.CH_UNK = self.chinese_stoi['<unk>']
        self.CH_SOS = self.chinese_stoi['<sos>']
        self.CH_EOS = self.chinese_stoi['<eos>']
        self.CH_PAD = self.chinese_stoi['<pad>']

        self.english_vocab_size = len(self.english_stoi)
        self.chinese_vocab_size = len(self.chinese_stoi)

    def chinese_2_vector(self, sentence):
        words = chinese_tokenizer(sentence)
        if len(words) >= self.chinese_fix_length:
            words = words[:self.chinese_fix_length - 2]
        words.insert(0, self.CH_SOS)
        words.append(self.CH_EOS)
        # 填充
        words += [self.CH_PAD for _ in range(self.chinese_fix_length - len(words))]
        # 将索引序列转为 PyTorch Tensor
        return torch.LongTensor([self.chinese_stoi.get(word, self.CH_UNK) for word in words], device=self.device)

    def vector_2_english(self, vector, join=False):
        if isinstance(vector, torch.Tensor):
            vector = vector.tolist()
        # 使用vocab.itos将索引转为词语
        words = [self.english_itos[idx] for idx in vector]
        return english_detokenizer(words) if join else words

    def get_start_sequence(self):
        return [self.EN_SOS]
