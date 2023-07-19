import os
import numpy as np
import uproot as ur
import matplotlib.pyplot as plt
import pandas as pd
import copy
import sys
from load_data import load_data_as_dict
from pathlib import Path
import re


def create_dataset(root_filename="data/fmrate.root",
               tunix_name="unix_time",
               new_columns=[],
               save_format=None):
    filename_no_extension = Path(root_filename).stem
    # Load the whole root file, keep all features: "None"
    data_dict = load_data_as_dict(root_filename=root_filename,
                            TTree_features_dict={filename_no_extension: None})
    
    # "Flattening" so that we can later have a vector for each example
    keys = [k for k in data_dict.keys()]
    for key in keys: 
        if data_dict[key].ndim > 1:
            if data_dict[key].shape[1] < 2:
                # create key, value pair for each dim of "key" separately.
                # Combine that dictionary with data_dict
                data_dict |= {key: data_dict[key].flatten()}
            else:
                # create key, value pair for each dim of "key" separately.
                # Combine that dictionary with data_dict
                data_dict |= dict(zip([f"{key}[{i}]" for i in range(data_dict[key].shape[1])], data_dict[key].T))
                # Remove from data_dict, the key "key"
                del data_dict[key]
    
    del keys    
    
    print("\nAvailable feature names or target names from root file: ", data_dict.keys())
    data_df = pd.DataFrame.from_dict(data_dict)
    sample_spacing = int(data_df["unix_time"].iloc[1] - data_df["unix_time"].iloc[0])
    print("Sample spacing between examples: ", sample_spacing)
    # We don't bin the data anymore
    
    ### XXX: here apply some further preprocessing
    
    # Create new columns of data_df
    if len(new_columns) != 0:
        # data_df["rate[0]/rate_err[0]"] = data_df["rate[0]"]/data_df["rate_err[0]"]
        for col in new_columns:
            # you can try with col = 'rate + rate[0]/(rate_err_0[90] + 5.) + rate' (just for the regex)
            operands = re.findall('[\w]+[\[][0-9]*[\]]|[\w]+', col)
            expression = col
            for op in operands:
                expression.replace(op, f"data_df[{op}]")
            data_df[col] = eval(expression)
    
    print("\nAvailable feature names or target names from data_df: ", data_df.columns)
    
    
    if save_format:
        os.makedirs('data', exist_ok=True)
        if save_format == "csv":
            data_df.to_csv(f'data/{filename_no_extension}_dataset.csv', index=False)
        elif save_format == "pkl":
            pd.to_pickle(data_df, f'data/{filename_no_extension}_dataset.pkl')
        else:
            raise NotImplementedError(f"save_format {save_format} not supported."+\
                                      "Only supporting csv or pkl")
    return data_df

if __name__ == "__main__":
    #create_dataset("data/nf1rate.root")
    pass

