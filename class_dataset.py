import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self,index):
        sample = {'X': self.X[index], 'y': self.y[index]}
        return sample