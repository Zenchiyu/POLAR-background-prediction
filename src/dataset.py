import pandas as pd
import torch
from torch.utils.data import Dataset
from create_dataset import create_dataset


class PolarDataset(Dataset):
    def __init__(self,
                 filename, 
                 feature_names,
                 target_names,
                 device="cpu",
                 transform=None,
                 target_transform=None,
                 save_format=None):
        
        super(PolarDataset, self).__init__()
        
        # Full dataset including targets
        if filename.endswith(".root"):
            # Create dataset file if cfg.dataset.target_format is not None
            self.data_df = create_dataset(filename, save_format)
        elif filename.endswith(".pkl"):
            self.data_df = pd.read_pkl(filename)
        elif filename.endswith(".csv"):
            self.data_df = pd.read_csv(filename)
        else:
            raise NotImplementedError(f"Extension of {filename} not supported."+\
                                      "Only supporting csv, pkl or root")
            
        self.X = torch.tensor(self.data_df[feature_names].values,
                              dtype=torch.float,
                              device=device)
        self.y = torch.tensor(self.data_df[target_names].values,
                              dtype=torch.float,
                              device=device)
        
        self.n_examples = self.X.shape[0]
        self.n_features = self.X.shape[1]
        self.n_targets = self.y.shape[1] if self.y.dim() > 1 else 1
        
        self.feature_names = feature_names
        self.id2feature_names = self.feature_names
        self.feature_names2id = {f: i for i, f in enumerate(feature_names)}
        
        self.target_names = target_names
        self.id2target_names = self.target_names
        self.target_names2id = {t: i for i, t in enumerate(target_names)}
        
        
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
    
