# -*- coding: utf-8 -*-
# @Time    : 2023/9/8 16:46
# @Author  : Liang Jinaye
# @File    : 01_Dataset.py
# @Description :
import os

from torch.utils.data import Dataset

from PIL import Image

DATASET_PATH = "E:\\Datasets\\hymenoptera_data\\train"


class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


if __name__ == "__main__":
    gb_root_dir = DATASET_PATH
    ants_label_dir = "ants"
    bees_label_dir = "bees"
    ants_dataset = MyData(gb_root_dir, ants_label_dir)
    bees_dataset = MyData(gb_root_dir, bees_label_dir)
    #
    train_dataset = ants_dataset + bees_dataset

    print(len(train_dataset))
