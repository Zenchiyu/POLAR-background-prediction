import pandas as pd
import torch
from torch.utils.data import Dataset


class PolarDataset(Dataset):
    def __init__(self,
                 filename, 
                 feature_names,
                 target_names,
                 device="cpu",
                 transform=None,
                 target_transform=None):
        
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
        
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        features = self.X[idx]
        targets = self.y[idx]
        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            targets = self.target_transform(targets)

        return features, targets
    
