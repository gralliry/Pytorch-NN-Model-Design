#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/29 0:07
# @Author  : Jianye Liang
# @File    : draft.py
# @Description :
import json
import os
import csv

ROOT_PATH = "E:\\Datasets\\translation2019zh"

with open(os.path.join(ROOT_PATH, "sentences_valid.tsv"), "w", newline='', encoding='utf-8') as tsv_file:
    fieldnames = ['english', 'chinese']
    writer = csv.DictWriter(tsv_file, fieldnames=fieldnames, delimiter='\t')  # 使用制表符作为分隔符
    # 写入 TSV 文件的头部
    writer.writeheader()

    with open(os.path.join(ROOT_PATH, "translation2019zh_valid.json"), "r", encoding='utf-8') as file:
        for line in file:
            jsonobj = json.loads(line.strip())
            writer.writerow({'english': jsonobj['english'], 'chinese': jsonobj['chinese']})
