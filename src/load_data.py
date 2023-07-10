import numpy as np
import uproot as ur

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
    
