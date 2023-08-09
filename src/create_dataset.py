import numpy as np
import os
import pandas as pd
import re
import typing

from pathlib import Path
from typing import Optional
from utils import generator_expressions, load_data_as_dict


def create_dataset(root_filename: str = "data/fmrate.root",
                   new_columns: list[str] = [],
                   filter_conditions: list[str] = [],
                   save_format: Optional[str] = None) -> pd.DataFrame:
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
    
    Order on these new column names can matter (e.g creating a new column
    based on a new column).

    - Can also filter based on existing columns (incl. new columns after their
    creation via new_columns).
    The filters are specified as a list of string where each string must contain
    operations over existing dataframe column names.

    They should be expressions that, when evaluated, return some boolean value
    (after replacing the column names by the corresponding array/dataframe)
    
    Order on these filters matters.
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
    
    # Further preprocessing
    # Create new columns of data_df
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
    
    One new (key, value) for each column of a 2d matrix (value is a vector),
    for each 2d matrix.
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

    if len(new_columns) == 0:
        return None
    
    expressions = generator_expressions(new_columns)

    for col, expression in zip(new_columns, expressions):
        if verbose: print(f"\nExpr to eval for col {col}: {expression}")
        eval(expression)  # just to show any warnings or errors

        # Add new column with evaluated expression
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
    Filter based on existing columns (incl. new columns after their
    creation via new_columns).
    The filters are specified as a list of string where each string must contain
    operations over existing dataframe column names.

    They should be expressions that, when evaluated, return some boolean value
    (after replacing the column names by the corresponding array/dataframe)
    
    Order on these filters matters.
    """

    if len(filter_conditions) == 0:
        return None

    expressions = generator_expressions(filter_conditions)

    for cond, expression in zip(filter_conditions, expressions):
        if verbose: print(f"\nExpr to eval for cond {cond}: {expression}")
        eval(expression)  # just to show any warnings or errors
        n_examples_old = data_df.shape[0]

        # Filter examples based on the condition, if true -> keep
        index_to_keep = data_df.loc[eval(expression.replace(".values", "")), :].index
        data_df.drop(index=data_df.index.difference(index_to_keep),
                        inplace=True)
        
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
            

