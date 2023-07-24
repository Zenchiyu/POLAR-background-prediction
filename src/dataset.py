import pandas as pd
import torch
import torchvision
import typing

from create_dataset import create_dataset
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional, TypeAlias


class PolarDataset(Dataset):
    def __init__(self,
                 filename: str, 
                 feature_names: list[str],
                 target_names: list[str],
                 device: str = "cpu",
                 transform: Optional[torch.nn.Module]=None,
                 target_transform: Optional[torch.nn.Module]=None,
                 new_columns: list[str] = [],
                 save_format: Optional[str] = None) -> None:
        
        super(PolarDataset, self).__init__()
        
        # Full dataset including targets
        match Path(filename).suffix:
            case ".root":
                # Create dataset file if cfg.dataset.target_format is not None
                self.data_df = create_dataset(filename, 
                                              new_columns=new_columns,
                                              save_format=save_format)
            case ".pkl":
                self.data_df = pd.read_pickle(filename)
            case ".csv":
                self.data_df = pd.read_csv(filename)
            case _:
                raise NotImplementedError(f"Extension {Path(filename).suffix} of {filename} not supported."+\
                                          "Only supporting csv, pkl or root")
        
        self.X = torch.tensor(self.data_df[feature_names].values.astype(float),
                              dtype=torch.float,
                              device=device)
        self.y = torch.tensor(self.data_df[target_names].values.astype(float),
                              dtype=torch.float,
                              device=device)
        
        self.n_examples: int = self.X.shape[0]
        self.n_features: int = self.X.shape[1]
        self.n_targets: int = self.y.shape[1] if self.y.dim() > 1 else 1
        
        self.feature_names = feature_names
        self.id2feature_names = self.feature_names
        self.feature_names2id = {f: i for i, f in enumerate(feature_names)}
        
        self.target_names = target_names
        self.id2target_names = self.target_names
        self.target_names2id = {t: i for i, t in enumerate(target_names)}
        
        
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.X[idx]
        targets = self.y[idx]
        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            targets = self.target_transform(targets)

        return features, targets
    
