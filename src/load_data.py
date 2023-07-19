import numpy as np
import uproot as ur
import sklearn
from sklearn.model_selection import train_test_split


def load_TTree(root_filename="../data/Allaux_Bfield.root",
                       TTree_name="t_hk_obox",
                       verbose=True):
    # Open the file and show its contents (like .ls in ROOT CERN)
    data = ur.open(root_filename+":"+TTree_name)
    if verbose:
        print(f"TTree: {TTree_name}'s contents:")
        data.show()
        print()
    return data

def load_data_as_dict(root_filename="../data/Allaux_Bfield.root",
			TTree_features_dict={
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
                      verbose=True):

    data_dict = {}
    for TTree_name, features_name in TTree_features_dict.items():
        data = load_TTree(root_filename=root_filename,
			  TTree_name=TTree_name)
        data_dict |= data.arrays(features_name, library="np")
    return data_dict
    


def train_val_test_split(X, y, val_size=0.2, test_size=0.2, random_state=42, shuffle=True):
    """
    Split the dataset into train, validation and test sets.
    X and y are pandas dataframes
    """
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