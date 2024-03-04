#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :
import pandas as pd


class Table:
    def __init__(self, path, sep=',', columns: list = None):
        self.table = pd.read_csv(path, sep=sep, names=columns)

    def show(self):
        print(self.table)


if __name__ == "__main__":
    table = Table("E:\\Datasets\\ImageNet\\meta\\alllabel.tsv", sep="\t")
    table.show()
