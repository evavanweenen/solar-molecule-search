import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

class SmilesDataset(Dataset):
    def __init__(self, filename, split, scaler=None, **kws_split):
        self.split = split

        self.X = pd.read_csv(f'{filename}_features.csv', index_col=0)
        if split != 'test':
            self.Y = pd.read_csv(f'{filename}_labels.csv', index_col=0)
        
        self.S = self.X.smiles

        self.X = self.X.drop('smiles', axis=1)

        if split == 'train':
            self.X, _, self.Y, _, self.S, _ = train_test_split(self.X, self.Y, self.S, **kws_split)
        elif split == 'val':
            _, self.X, _, self.Y, _, self.S = train_test_split(self.X, self.Y, self.S, **kws_split)

        self.X = torch.tensor(self.X.values).float()
        if split != 'test':
            self.Y = torch.tensor(self.Y.values).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.split != 'test':
            return self.X[idx], self.Y[idx]
        else:
            return self.X[idx]