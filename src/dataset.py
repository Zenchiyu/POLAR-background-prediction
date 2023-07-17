import pandas as pd
import torch
from torch.utils.data import Dataset


class PolarDataset(Dataset):
    def __init__(self,
                 filename, 
                 feature_names,
                 target_names,
                 device="cpu"):
        super(PolarDataset, self).__init__()
        # Full dataset including targets
        self.data_df = pd.read_csv(filename)
        self.X = torch.tensor(self.data_df[feature_names].values,
                              dtype=torch.float,
                              device=device)
        self.y = torch.tensor(self.data_df[target_names].values,
                              dtype=torch.float,
                              device=device)
        
        self.n_examples = self.X.shape[0]
        self.n_features = self.X.shape[1]
        self.n_targets = self.y.shape[1] if self.y.dim() > 1 else 1
        
    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
