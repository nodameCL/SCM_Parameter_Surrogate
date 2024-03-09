from torch.utils.data import Dataset
import pandas as pd 
import numpy as np
import sys
import torch
import torch.utils.data
import os 
from ast import literal_eval

class SurfaceComplexationDataset(Dataset):
    """ surface complexation dataset """
    def __init__(self, 
                 root_dir, 
                 split = 'train'): 
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.root = root_dir 
        self.csv_file = os.path.join(root_dir, '{}_flatten.csv'.format(split))
        self.data = pd.read_csv(self.csv_file)

        # get inputs and targets 
        self.x = self.data.drop(['c1', 'c2', 'logk1', 'logk2', 'logKc', 'logKa'], axis = 1)
        self.y = self.data[['c1', 'c2', 'logk1', 'logk2', 'logKc', 'logKa']]

        self.x = torch.from_numpy(np.array(self.x).astype(np.float32))
        self.y = torch.from_numpy(np.array(self.y).astype(np.float32))

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, index): 
        return self.x[index], self.y[index]


if __name__ == '__main__':
    
    d = SurfaceComplexationDataset(root_dir = data_dir, split='test')
    print("length of dataset is:", len(d), type(d))
   
    x, y = d[0] # get item when index = 0 
    print("shape of input: ", x.shape, type(x), x)
    print("shape of output: ", y.shape, type(y),  y)