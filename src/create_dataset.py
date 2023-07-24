import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import sys
import typing
import uproot as ur

from utils import load_data_as_dict
from pathlib import Path
from typing import Optional


def create_dataset(root_filename: str = "data/fmrate.root",
                   tunix_name: str = "unix_time",
                   new_columns: list[str] = [],
                   save_format: Optional[str] = None,
                   filter_conditions: list[str] = []) -> pd.DataFrame:
    """
    - Create pandas dataframe containing the whole dataset (features & target) from
    a .root file.
    
    - Optionally save dataframe into specified format among these:
        - "pkl"
        - "csv"
        - None (when don't want to save)
    
    - Can also create new columns based on existing columns.
    Each new column is specified by a string in new_columns list. The string
    must contain operations over existing dataframe column names.
    """
    filename_no_extension = Path(root_filename).stem
    
    # Load the whole root file while keeping all features:
    # filename_no_extension: None -> keeps all features
    data_dict = load_data_as_dict(root_filename=root_filename,
                            TTree_features_dict={filename_no_extension: None})
    
    # "Flattening" so that we can later have a vector for each example
    flatten_dict_arrays(data_dict)
    
    # Create pandas dataframe from dictionary
    data_df = pd.DataFrame.from_dict(data_dict)
    
    # We don't bin the data anymore
    
    ### XXX: Apply some further preprocessing
    # Create new columns of data_df and evaluate expressions
    create_new_columns(data_df, new_columns=new_columns)
    # Filter some examples based on "filter" (true -> keep)
    filter_examples(data_df, filter_conditions=filter_conditions)
    
    
    sample_spacing = int(data_df["unix_time"].iloc[1] - data_df["unix_time"].iloc[0])
    print("Sample spacing (seconds) between examples: ", sample_spacing)
    print("\nAvailable feature/target names from root file: ", data_dict.keys())
    print("\nAvailable feature/target names from data_df: ", data_df.columns)
    print(data_df.dtypes)
    
    if save_format: df_save_format(data_df, filename_no_extension, save_format)
    
    return data_df

def flatten_dict_arrays(data_dict: dict[str, np.typing.NDArray[typing.Any]]) -> None:
    """
    "Flattening" each matrix from data_dict by distributing its columns into
    new keys.
    
    One new (key, value) for each column of a 2d matrix, for each 2d matrix
    where the value is a vector.
    """
    keys = [k for k in data_dict.keys()]
    for key in keys:
        # create key, value pair for each dim of "key" separately.
        # Combine that dictionary with data_dict
        if data_dict[key].ndim > 1:
            if data_dict[key].shape[1] < 2:
                data_dict |= {key: data_dict[key].flatten()}
            else:
                # Create multiple keys
                new_keys = [f"{key}[{i}]" for i in range(data_dict[key].shape[1])]
                data_dict |= dict(zip(new_keys, data_dict[key].T))
                
                del data_dict[key]   
    del keys

def create_new_columns(data_df: pd.DataFrame,
                       new_columns: list[str] = [],
                       verbose: bool = True) -> None:
    """
    Create new columns of data_df. Can evaluate expressions to create
    columns based on existing columns.
    
    Warning: careful with "silent" errors related to the operations on existing
    columns (e.g division by 0)
    """

    if len(new_columns) != 0:
        for col in new_columns:
            operands = re.finditer('[\w]+[\[][0-9]*[\]]', col)
            expression = col
            incr = 0
            for op in operands:
                start_idx, end_idx = op.span()
                before = expression[:start_idx+incr]
                after = expression[end_idx+incr:]
                expression = before + f"data_df['{op.group()}'].values" + after
                incr += len("data_df[''].values")
            if verbose: print(f"Expr to eval for col {col}: {expression}")
            
            # Add new column with evaluated expression
            eval(expression)  # just to show any warnings or errors
            data_df[col] = eval(expression.replace(".values", ""))
            
            n_examples_old = data_df.shape[0]
            # Filter out the rows having at least a NaN or missing value
            data_df.dropna(inplace=True)
            
            if verbose:
                print("Number of examples before filtering: ", n_examples_old)
                print("Number of examples after filtering (if happened): ", data_df.shape[0])
    
def filter_examples(data_df: pd.DataFrame,
                    filter_conditions: list[str] = [],
                    verbose: bool = True) -> None:
    """
    """

    if len(filter_conditions) != 0:
        for col in filter_conditions:
            operands = re.finditer('[\w]+[\[][0-9]*[\]]', col)
            expression = col
            incr = 0
            for op in operands:
                start_idx, end_idx = op.span()
                before = expression[:start_idx+incr]
                after = expression[end_idx+incr:]
                expression = before + f"data_df['{op.group()}'].values" + after
                incr += len("data_df[''].values")
            if verbose: print(f"Expr to eval for col {col}: {expression}")
            
            # Add new column with evaluated expression
            eval(expression)  # just to show any warnings or errors
            n_examples_old = data_df.shape[0]

            # Filter examples based on the condition, if true -> keep
            data_df = data_df[eval(expression.replace(".values", ""))]
            # Filter out the rows having at least a NaN or missing value
            data_df.dropna(inplace=True)
            
            if verbose:
                print("Number of examples before filtering: ", n_examples_old)
                print("Number of examples after filtering (if happened): ", data_df.shape[0])


def df_save_format(data_df: pd.DataFrame,
                   filename_no_extension: str = "fmrate",
                   save_format: Optional[str] = None) -> None:
    print(f"Saving dataset in {save_format} format")
    os.makedirs('data', exist_ok=True)
    match save_format:
        case "csv":
            data_df.to_csv(f'data/{filename_no_extension}_dataset.csv', index=False)
        case "pkl":
            data_df.to_pickle(f'data/{filename_no_extension}_dataset.pkl')
        case _:
            raise NotImplementedError(f"save_format {save_format} not supported."+\
                                  "Only supporting csv or pkl")
            

