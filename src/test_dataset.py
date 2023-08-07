import pandas as pd
import torch
import numpy as np

from create_dataset import create_dataset
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Optional


class MyDataset(Dataset):
    def __init__(self,
                 feature_names: list[str],
                 target_names: list[str]) -> None:
        
        super(MyDataset, self).__init__()
        arr = np.arange(10).reshape(5, 2)
        self.data_df = pd.DataFrame(arr, columns=["x", "y"])
        # print(self.data_df)
        
        # self.X = torch.tensor(self.data_df[feature_names].values.astype(float),
        #                       dtype=torch.float,
        #                       device=device)
        # self.y = torch.tensor(self.data_df[target_names].values.astype(float),
        #                       dtype=torch.float,
        #                       device=device)
        
        self.n_examples: int = self.data_df.shape[0]
        self.n_features: int = self.data_df.shape[1]
        self.n_targets: int = len(target_names)
        
        self.feature_names = feature_names
        self.id2feature_names = self.feature_names
        self.feature_names2id = {f: i for i, f in enumerate(feature_names)}
        
        self.target_names = target_names
        self.id2target_names = self.target_names
        self.target_names2id = {t: i for i, t in enumerate(target_names)}
    
    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.data_df[self.feature_names][idx].values.astype(float)
        targets = self.data_df[self.target_names][idx].values.astype(float)
        return features, targets
    
if __name__ == "__main__":
    d = MyDataset(["x"], ["y"])
    # print(d[:][0])
    loader = DataLoader(d, batch_size=None)
    print(loader.dataset[:])