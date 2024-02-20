#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/4 23:15
# @Author  : Gralliry
# @File    : test.py
# @Description :
import csv

import torch

from dataset import MyDataset
from model import TextRCNN

CUDA = True


def main():
    device = torch.device('cuda' if torch.cuda.is_available() and CUDA else 'cpu')

    dataset = MyDataset(device=device, batch_size=2048)
    # 实例化模型
    run_model = TextRCNN(len(dataset.text.vocab)).to(device)
    run_model.load_state_dict(
        torch.load("parameter/TextRCNN_2024-02-08-19-24-17_9_0.670959887222863.pth", map_location=device))

    data = [["PhraseId", "Sentiment"]]
    phraseId_data = []
    sentiment_data = []
    run_model.eval()
    with torch.no_grad():
        for phraseId, text in dataset.get_test_iterator():
            predictions = run_model(text)
            phraseId_data.extend(phraseId.tolist())
            sentiment_data.extend(predictions.argmax(dim=1).tolist())
    data.extend([[i, s] for i, s in zip(phraseId_data, sentiment_data)])
    # 写入 CSV 文件
    with open("test_result.csv", 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # 写入数据
        csvwriter.writerows(data)


if __name__ == "__main__":
    main()
