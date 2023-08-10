import pandas as pd
import torch

from create_dataset import create_dataset
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional


class PolarDataset(Dataset):
    def __init__(self,
                 filename: str, 
                 feature_names: list[str],
                 target_names: list[str],
                 transform: Optional[torch.nn.Module]=None,
                 target_transform: Optional[torch.nn.Module]=None,
                 new_columns: list[str] = [],
                 filter_conditions: list[str] = [],
                 save_format: Optional[str] = None) -> None:
        
        super(PolarDataset, self).__init__()
        
        # Full dataset including targets
        match Path(filename).suffix:
            case ".root":
                # Save dataset file if cfg.dataset.save_format is not None
                data_df = create_dataset(filename, 
                                              new_columns=new_columns,
                                              filter_conditions=filter_conditions,
                                              save_format=save_format)
            case ".pkl":
                data_df = pd.read_pickle(filename)
            case ".csv":
                data_df = pd.read_csv(filename)
            case _:
                raise NotImplementedError(f"Extension {Path(filename).suffix} of {filename} not supported."+\
                                          "Only supporting csv, pkl or root")
        data_np = data_df.values.astype(float)
        X_np = data_df[feature_names].values.astype(float)
        y_np = data_df[target_names].values.astype(float)
        
        # Tensors
        self.data_cpu = torch.tensor(data_np,
                                     dtype=torch.float,
                                     device="cpu")
        self.X_cpu = torch.tensor(X_np,
                                  dtype=torch.float,
                                  device="cpu")
        self.y_cpu = torch.tensor(y_np,
                                  dtype=torch.float,
                                  device="cpu")
        
        # Shapes/sizes
        self.n_examples: int = X_np.shape[0]
        self.n_features: int = X_np.shape[1]
        self.n_targets: int = len(target_names)

        # Column names of data_cpu
        self.column_names = list(data_df.columns)
        self.id2column_names = self.column_names
        self.column_names2id = {c: i for i, c in enumerate(data_df.columns)}
        
        # Features
        self.feature_names = feature_names
        self.id2feature_names = self.feature_names
        self.feature_names2id = {f: i for i, f in enumerate(feature_names)}
        
        # Targets
        self.target_names = target_names
        self.id2target_names = self.target_names
        self.target_names2id = {t: i for i, t in enumerate(target_names)}
        
        # Transforms
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        # Data on CPU
        features = self.X_cpu[idx]
        targets = self.y_cpu[idx]

        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            targets = self.target_transform(targets)

        return features, targets, idx
    
