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