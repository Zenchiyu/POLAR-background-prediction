import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import sys
import uproot as ur

from load_data import load_data_as_dict
from pathlib import Path


def create_dataset(root_filename="data/fmrate.root",
               tunix_name="unix_time",
               new_columns=[],
               save_format=None):
    filename_no_extension = Path(root_filename).stem
    
    # Load the whole root file, keep all features: "None"
    data_dict = load_data_as_dict(root_filename=root_filename,
                            TTree_features_dict={filename_no_extension: None})
    
    # "Flattening" so that we can later have a vector for each example
    flatten_dict_arrays(data_dict)
    
    # Create pandas dataframe from dictionary
    data_df = pd.DataFrame.from_dict(data_dict)
    
    # We don't bin the data anymore
    
    ### XXX: here apply some further preprocessing
    # Create new columns of data_df and evaluate expressions
    if len(new_columns) != 0:
        for col in new_columns:
            operands = re.finditer('[\w]+[\[][0-9]*[\]]|[\w]+', col)
            expression = col
            incr = 0
            for op in operands:
                start_idx, end_idx = op.span()
                before = expression[:start_idx+incr]
                after = expression[end_idx+incr:]
                expression = before + f"data_df['{op.group()}']" + after
                incr += len(f"data_df['']")
            print(f"Expr to eval for col {col}: {expression}")
            data_df[col] = eval(expression)
    
    
    sample_spacing = int(data_df["unix_time"].iloc[1] - data_df["unix_time"].iloc[0])
    print("Sample spacing (seconds) between examples: ", sample_spacing)
    print("\nAvailable feature names or target names from root file: ", data_dict.keys())
    print("\nAvailable feature names or target names from data_df: ", data_df.columns)
    print(data_df.dtypes)
    
    if save_format: df_save_format(data_df, filename_no_extension, save_format)
    
    return data_df

def flatten_dict_arrays(data_dict):
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
                data_dict |= dict(zip([f"{key}[{i}]" for i in range(data_dict[key].shape[1])], data_dict[key].T))
                # Remove from data_dict, the key "key"
                del data_dict[key]   
    del keys 

def df_save_format(data_df, filename_no_extension="fmrate", save_format=None):
    os.makedirs('data', exist_ok=True)
    if save_format == "csv":
        data_df.to_csv(f'data/{filename_no_extension}_dataset.csv', index=False)
    elif save_format == "pkl":
        data_df.to_pickle(f'data/{filename_no_extension}_dataset.pkl')
    else:
        raise NotImplementedError(f"save_format {save_format} not supported."+\
                                  "Only supporting csv or pkl")
            

