import numpy as np
import pandas as pd
import torch
import typing
import uproot as ur

from torch.utils.data import Subset
from typing import Optional
from numpy.typing import NDArray


def load_TTree(root_filename: str = "../data/Allaux_Bfield.root",
               TTree_name: str = "t_hk_obox",
               verbose: bool = True) -> ur.reading.ReadOnlyDirectory:
    # Open the file and show its contents (like .ls in ROOT CERN)
    data = ur.open(root_filename+":"+TTree_name)
    if verbose:
        print(f"TTree: {TTree_name}'s contents:")
        data.show()
        print()
    return data

def load_data_as_dict(root_filename: str = "../data/Allaux_Bfield.root",
                      TTree_features_dict: dict[str, Optional[list[str]]] = {
                        "t_hk_obox":
                                ["saa", 
                               "raz",
                               "decz",
                               "rax",
                               "decx",
                               "obox_mode",
                               "fe_temp",
                               "glon", 
                               "glat",
                               "tunix",
                               "fe_cosmic",
                               "fe_rate"],
                        },
                      verbose: bool = True) -> dict[str, NDArray[typing.Any]]:

    data_dict: dict[str, NDArray[typing.Any]] = {}
    for TTree_name, features_name in TTree_features_dict.items():
        data = load_TTree(root_filename=root_filename,
			  TTree_name=TTree_name)
        data_dict |= data.arrays(features_name, library="np")
    return data_dict
    


def train_val_test_split(X: pd.DataFrame,
                         y: pd.DataFrame,
                         val_size: float =0.2,
                         test_size: float =0.2,
                         random_state: Optional[int] =42,
                         shuffle: bool = True) -> tuple[pd.DataFrame, ...]:
    """
    Split the dataset into train, validation and test sets.
    X and y are pandas dataframes
    """
    import sklearn
    from sklearn.model_selection import train_test_split

    assert val_size + test_size < 1, "There's no training examples, need some training examples" 
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size, 
                                                        random_state=random_state, 
                                                        shuffle=shuffle)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                      test_size=val_size/(1-test_size), 
                                                      random_state=random_state,
                                                      shuffle=shuffle) 
    # 1/4 = 20/80 = 0.2/(0.2+0.6)
    # Reset indices of df:
    X_train.reset_index(drop=True, inplace=True)
    X_val.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def merge_torch_subsets(subsets: list[torch.utils.data.Subset]):
    """
    Merge PyTorch Subsets assuming the underlying dataset is the same.
    Will merge indices
    """
    indices = list(set().union(*[subset.indices for subset in subsets]))
    return Subset(subsets[0].dataset, indices)