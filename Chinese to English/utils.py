#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :

def chinese_tokenizer(sentence: str):
    return sentence.split()


def chinese_detokenizer(words: list):
    return "".join(words)


def english_tokenizer(sentence: str):
    return sentence.split(" ")


def english_detokenizer(words: list):
    return " ".join(words)


def save_vocab(vocab, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(str(vocab))


def load_stoi(save_path):
    with open(save_path, 'r', encoding='utf-8') as f:
        dic = f.read()
        dic = list(dic)
        for i in range(len(dic)):
            if dic[i] == '{':
                del dic[0:i]
                del dic[-1]
                break
        dic = ''.join(dic)
        dic = eval(dic)
    return dic


def load_itos(save_path):
    with open(save_path, 'r', encoding='utf-8') as f:
        dic = f.read()
        dic = eval(dic)
    return dic
