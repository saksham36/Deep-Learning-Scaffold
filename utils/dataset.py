#!/usr/bin/env python3.8
'''
Written by: Saksham Consul 04/28/2023
Scripts needed for creating dataloaders
'''
import os
import numpy as np
import pandas as pd
from enum import Enum
from utils.filtering import msv

import torch
from torch.utils.data import Dataset


class Actions(Enum):
    REST = 0
    FLEX = 1
    EXT = 2
    RAD = 3
    ULN = 4
    SUP = 5
    PRO = 6
    PAD = -1


class sEMGDataset(Dataset):
    """Custom Dataset for sEMG signals"""

    def __init__(self, args):
        filename = os.path.join(args.data_path, args.data_file)

        self.file = pd.read_pickle(filename)
        sentence_column = self.file['sentence']
        self.data = torch.from_numpy(
            np.stack(sentence_column.to_numpy())).float()

        label_column = self.file['label']
        self.labels = np.stack(label_column.to_numpy())

        # (n_sentences, n_words, n_classes)
        self.y = torch.from_numpy(self.one_hot_encoding(self.labels)).float()

    def __getitem__(self, index):
        x = self.data[index, ...]

        label = self.y[index, ...]

        return x, label

    def __len__(self):
        return self.y.shape[0]

    def one_hot_encoding(self, labels):
        """
        one hot encoding for labels
        """
        n_sentences, n_words = labels.shape
        n_unique_labels = len(Actions)
        one_hot_encode = np.zeros((n_sentences, n_words, n_unique_labels))
        # One hot encode the labels
        for i in range(n_sentences):
            for j in range(n_words):
                one_hot_encode[i, j, labels[i, j].astype(int)] = 1
        return one_hot_encode
