import gc
import numpy as np
import pandas as pd
import re
import torch
import typing
import uproot as ur

from numpy.typing import NDArray
from torch.utils.data import Dataset, Subset
from typing import Generator, Optional, Callable, Any


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
                          TTree_name=TTree_name,
                          verbose=verbose)
        data_dict |= data.arrays(features_name, library="np")
    return data_dict

def periodical_split(dataset: Dataset,
                     percentages: list[float],
                     periodicity: int) -> tuple[Subset, ...]:
    """
    Periodically split a dataset into non-overlapping new datasets.

    Example:
    If we use it to obtain a train, validation and test set,
    the resulting split from the dataset would look like:
    train, val, test, train, val, test, train, val, test etc.

    So, if we have:
    percentages = [0.6, 0.2, 0.2] (for train, val, test sets)
    periodicity = 10

    then the different PyTorch Subsets contain data points
    from these ranges:
    - 0:6,  10:16, 20:26 etc. (train)
    - 6:8,  16:18, 26:28 etc. (val)
    - 8:10, 18:20, 28:30 etc. (test)
    """
    assert sum(percentages) == 1, "lengths should sum to 1"

    all_indices = np.arange(len(dataset))
    ns = [int(l*periodicity) for l in percentages]
    cumsum = np.cumsum([0] + ns)

    indices = [np.nonzero(np.isin((all_indices % periodicity),
                                  np.arange(begin, end)))[0] for begin, end in zip(cumsum[:-1], cumsum[1:])]
    subsets = tuple([Subset(dataset, idxs) for idxs in indices])
    return subsets

def merge_torch_subsets(subsets: list[torch.utils.data.Subset]) -> Subset:
    """
    Merge PyTorch Subsets assuming the underlying dataset is the same.
    Will merge indices
    """
    indices = list(set().union(*[subset.indices for subset in subsets]))
    return Subset(subsets[0].dataset, indices)

def delete(el: Any) -> None:
    try:
        el.to(device="cpu")
    except:
        print(f"Element of type {type(el)} you're trying to "+\
              "delete doesn't have a .to() method "+\
                "so changing its device to CPU is ignored")
    del el
    torch.cuda.empty_cache()
    gc.collect()

def generator_expressions(raw_expressions: list[str] = []) -> Generator[str, None, None]:
    # Generate expressions that can be evaluated from the raw_expressions
    for raw_expr in raw_expressions:
        # Match a column name of the form:
        # - column_64_name[some number]
        # or
        # - column_64_name
        operands = re.finditer('[\w]+[\[][0-9]*[\]]|[a-zA-Z][\w]*', raw_expr)
        expression = raw_expr
        incr = 0
        for op in operands:
            start_idx, end_idx = op.span()
            before = expression[:start_idx+incr]
            after = expression[end_idx+incr:]
            expression = before + f"data_df['{op.group()}'].values" + after
            incr += len("data_df[''].values")
        
        yield expression
    
def numpy_output(func: Callable[..., NDArray[Any]|torch.Tensor|
    tuple[NDArray[Any]|torch.Tensor, ...]]) -> Callable[..., NDArray[Any]|tuple[NDArray[Any], ...]]:
    # Used as decorator to convert tensors into numpy arrays
    def wrapper_numpy(*args: Any, **kwargs: Any) -> NDArray[Any]|tuple[NDArray[Any], ...]:
        res = func(*args, **kwargs)
        if type(res) == type(tuple()):
            return tuple(map(np.array, res))
        return np.array(res)
    return wrapper_numpy