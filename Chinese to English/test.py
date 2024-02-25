#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :

import torch

from model import Model
from dataset import TestDataset


def translate(sentence, model, dataset, device):
    # batch_size = 1 # Convert to Tensor
    words = dataset.chinese_2_vector(sentence)

    outputs = dataset.get_start_sequence()
    model.eval()
    with torch.no_grad():
        for i in range(dataset.chinese_fix_length):
            trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

            output = model(words, trg_tensor)

            best_guess = output.argmax(2)[-1, :].item()

            outputs.append(best_guess)

            if best_guess == dataset.EN_EOS:
                break
    # 英语索引翻译成英文
    return dataset.vector_2_english(outputs, join=True)


def main():
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 数据集
    dataset = TestDataset(device=device)
    print("数据集加载完成")

    embedding_size = 256
    num_heads = 4
    num_encoder_layers = 3
    num_decoder_layers = 3
    dropout = 0.1
    max_len = 30  # 最长一个句子的长度也不能超过 max_len

    forward_expansion = 512  # pytorch官方实现的transformer中，这个参数就是线性层升维后的结果

    # 模型 # Initialize network
    model = Model(
        embedding_size=embedding_size,
        src_vocab_size=dataset.chinese_vocab_size,
        trg_vocab_size=dataset.english_vocab_size,
        src_pad_idx=dataset.CH_PAD,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        forward_expansion=forward_expansion, dropout=dropout, max_len=max_len, device=device
    ).to(device)

    print("模型加载完成")

    while True:
        sentence = input("请输入中文: ")
        print("译文:", translate(sentence, model, dataset, device))


if __name__ == "__main__":
    main()
