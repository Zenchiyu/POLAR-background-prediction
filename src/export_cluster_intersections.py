import hydra
import numpy as np
import pandas as pd
import re
import torch
import uproot as ur

from functools import reduce
from numpy.typing import NDArray
from omegaconf import DictConfig
from pathlib import Path
from visualizer import find_moments, get_columns, get_all_time_y_y_hat
from trainer import Trainer
from torch.utils.data import Subset
from typing import Any
from utils import delete


def get_clusters(data_df: pd.DataFrame,
                 var: NDArray[Any],
                 new_std: float,
                 target_name: str,
                 id: int,
                 k: float,
                 pred_below: bool = True,
                 discard_w: int = 0) -> tuple[Any, ...]:
    """
    Return useful information about clusters such as the start and ending times.
    """
    t = data_df["unix_time"].values
    t_flip = np.flip(t)

    ## Red points of interest. Id is used to select a particular output
    if pred_below:
        mask = var[:, id] > k*new_std[id]
    else:
        mask = var[:, id] < -k*new_std[id]

    # Time difference for masked data
    t_mask = data_df[mask]["unix_time"].values
    idx_mask = np.flatnonzero(mask)  # index from original timeline
    delta = np.fix(np.diff(t_mask))
    delta = np.insert(delta, 0, np.inf)
    
    # Time difference for masked flipped data
    t_flip_mask = np.flip(t_mask)
    idx_flip_mask = np.flatnonzero(np.flip(mask))  # index from flipped timeline
    delta_flip = np.fix(np.diff(t_flip_mask))
    delta_flip = np.insert(delta_flip, 0, -np.inf)
    # np.fix rounds to nearest integer to 0.

    ## Starting points of clusters
    # Indices of the start points according to original timeline
    old_idx_points_starts = idx_mask[delta > 1]
    if discard_w:
        # Filter: if discard_w data points before are still within discard_w seconds before, keep the cluster
        t_sec_before_start = t[old_idx_points_starts]-discard_w
        t_data_before_start = np.concatenate([np.array([t[0]-i for i in range(discard_w, 0, -1)]), t])[old_idx_points_starts]
        idx_points_starts = np.intersect1d(old_idx_points_starts[t_data_before_start.astype(int) >= t_sec_before_start.astype(int)],\
                                        old_idx_points_starts)
        cluster_keep_mask = np.isin(old_idx_points_starts, idx_points_starts)
    else:
        idx_points_starts = old_idx_points_starts

    ## Ending points of clusters
    # Indices of the end points according to flipped timeline
    old_idx_points_ends = idx_flip_mask[delta_flip < -1]
    if discard_w:
        # Keep the ends if the starts were kept (again flipped timeline)
        old_idx_points_ends = old_idx_points_ends[np.flip(cluster_keep_mask)]

        # Do something similar to "starts" but on flipped timeline.
        # Filter: if discard_w data points after are still within discard_w seconds after, keep the cluster
        t_sec_after_end = t_flip[old_idx_points_ends]+discard_w
        t_data_after_end = np.concatenate([np.array([t[0]+i for i in range(discard_w, 0, -1)]), t_flip])[old_idx_points_ends]
        idx_points_ends = np.intersect1d(old_idx_points_ends[t_data_after_end.astype(int) <= t_sec_after_end.astype(int)],\
                                        old_idx_points_ends)
        cluster_keep_mask = np.isin(old_idx_points_ends, idx_points_ends)

        # Indices but in ORIGINAL timeline
        old_idx_points_ends = np.flip((t.size - 1) - old_idx_points_ends)
        idx_points_ends = np.flip((t.size - 1) - idx_points_ends)
        
        # Keep the correct starts and ends
        idx_points_starts = idx_points_starts[np.flip(cluster_keep_mask)]
    else:
        # Indices of the end points according to original timeline
        idx_points_ends = np.flip((t.size - 1) - old_idx_points_ends)
    
    # Get the start and end times !
    starts = data_df["unix_time"].iloc[idx_points_starts]
    ends = data_df["unix_time"].iloc[idx_points_ends]
    
    # Recreate mask due to removed clusters
    new_mask = np.zeros_like(mask, dtype=bool)
    for begin, end in zip(idx_points_starts, idx_points_ends):
        new_mask[begin:end+1] = mask[begin:end+1]
    mask = new_mask

    # Recreate times due to removed clusters
    times = data_df[mask]["unix_time"]
    times = times.reset_index(drop=True)

    # Adding a label to each group/cluster
    idx_starts = idx_points_starts
    idx_ends = idx_points_ends
    groups = np.repeat(np.arange(len(idx_starts)), idx_ends - idx_starts + 1)
    
    # XXX: only for the red points of interest !
    data = data_df[mask].copy()
    data["group"] = groups
    data["target_id"] = id

    # Two more properties of clusters
    integrals = data.groupby("group")[target_name].sum().values
    lengths = ends.values-starts.values
    # Drop the column 'group'
    data.drop(columns=["group"], inplace=True)
    
    return mask, data, times, starts, ends, groups, integrals, lengths

def get_inter_clusters(data_df: pd.DataFrame,
                       df: pd.DataFrame,
                       inter_id_or_cond: str) ->  tuple[Any]:
    """
    Get different properties of cluster intersections (doesn't differ much from before)
    """
    # XXX: First thing that change compared to before
    # XXX: Only focus on data coming from same type of intersection
    match inter_id_or_cond.split():
        case ["#", "inter", *_]:
            # If inter_id_or_cond is actually not inter_id but is some condition
            # based on # inter:
            match_obj = re.search("# inter", inter_id_or_cond)
            data_inter = df[eval("df['# inter']" + inter_id_or_cond[match_obj.end():])]            
        case _:
            inter_id = inter_id_or_cond
            data_inter = df[df["inter_id"] == inter_id]
    
    # If I want to select all the red points from the first rate
    # data_inter = df[np.array(list(map(lambda x: "0" in x, df["inter_id"])))]
    
    # Compute starts, ends etc.
    t = data_df["unix_time"].values
    t_flip = np.flip(t)

    # XXX: This is the second thing that change compared to before
    mask = np.zeros_like(t, dtype=bool)
    mask[data_inter.index.values] = True
    
    # Time difference for masked data
    t_mask = data_df[mask]["unix_time"].values
    idx_mask = np.flatnonzero(mask)  # index from original timeline
    delta = np.fix(np.diff(t_mask))
    delta = np.insert(delta, 0, np.inf)
    
    # Time difference for masked flipped data
    t_flip_mask = np.flip(t_mask)
    idx_flip_mask = np.flatnonzero(np.flip(mask))  # index from flipped timeline
    delta_flip = np.fix(np.diff(t_flip_mask))
    delta_flip = np.insert(delta_flip, 0, -np.inf)
    # np.fix rounds to nearest integer to 0.

    ## Starting points of clusters
    # Indices of the start points according to original timeline
    old_idx_points_starts = idx_mask[delta > 1]
    idx_points_starts = old_idx_points_starts

    ## Ending points of clusters
    # Indices of the end points according to original timeline
    old_idx_points_ends = np.flip((t.size - 1) - idx_flip_mask[delta_flip < -1])
    idx_points_ends = old_idx_points_ends
    
    # Get the start and end times !
    starts = data_df["unix_time"].iloc[idx_points_starts]
    ends = data_df["unix_time"].iloc[idx_points_ends]
    
    # Recreate mask due to removed clusters
    new_mask = np.zeros_like(mask, dtype=bool)
    for begin, end in zip(idx_points_starts, idx_points_ends):
        new_mask[begin:end+1] = mask[begin:end+1]
    mask = new_mask

    # Recreate times due to removed clusters
    times = data_df[mask]["unix_time"]
    times = times.reset_index(drop=True)

    # Adding a label to each group/cluster
    idx_starts = idx_points_starts
    idx_ends = idx_points_ends
    groups = np.repeat(np.arange(len(idx_starts)), idx_ends - idx_starts + 1)
    
    # XXX: only for the red points of interest !
    data = data_df[mask].copy()
    data["group"] = groups
    
    # Two properties of clusters
    integrals = data.groupby("group")["rate[0]"].sum().values # TODO, change this
    lengths = ends.values-starts.values
    data.drop(columns=["group"], inplace=True)
    return mask, data, times, starts, ends, groups, integrals, lengths

def get_merged_df(cfg: DictConfig,
                  data_df: pd.DataFrame,
                  var: NDArray[Any],
                  new_std: float,
                  k: float,
                  pred_below: int,
                  discard_w: int):
    """
    Merge red points of interests from different energy bins
    """
    # Information about clusters for all target names
    out = [get_clusters(data_df, var, new_std, target_name,
                        id, k, pred_below=pred_below,
                        discard_w=discard_w) for id, target_name in enumerate(cfg.dataset.target_names)]
    _, data, _, _, _, _, _, _ = zip(*out)

    # Merge dataframes containing red points of interests from different energy bins
    # Multiple new columns will appear in 'df' due to the different 'target_id''s
    df = reduce(lambda df1,df2: df1.merge(df2, on=list(data_df.keys()), how="outer", sort=True), data)
    # https://stackoverflow.com/questions/42940507/merging-dataframes-keeping-all-items-pandas
    # https://stackoverflow.com/questions/38978214/merge-a-list-of-dataframes-to-create-one-dataframe
    
    # Put the index back
    df = df.set_index(np.flatnonzero(np.isin(data_df["unix_time"], df["unix_time"])))
    nb_inter = df.drop(columns=list(data_df.keys())).count(axis=1)
    
    # Fuse columns:
    df["inter_id"] = df.iloc[:, len(data_df.keys()):].apply(
        lambda x: ','.join(x.dropna().astype(int).astype(str)), axis=1
    )
    # See: https://stackoverflow.com/questions/33098383/merge-multiple-column-values-into-one-column-in-python-pandas
    # Drop columns from the merge:
    df = df[list(data_df.keys()) + ["inter_id"]]
    # Add '# inter' column:
    df["# inter"] = nb_inter
    return df

@hydra.main(version_base=None, config_path="../config", config_name="trainer")
def main(cfg: DictConfig):
    """
    - Export in two different .ROOT files, both the predictions and the cluster
    intersections.
    - Also export, in .pkl format the number of positive/negative examples or clusters
    as well as the number of positive/negative cluster intersections for different sets
    of energy bins or conditions.

    Here's a minimal code to load the two pickled dataframes (for export_last=True):
    ```
    import pandas as pd
    counts = pd.read_pickle("data/num_examples_clusters.pkl")
    counts_inter = pd.read_pickle("data/num_cluster_intersections.pkl")
    ```
    Please change the path accordingly if you're not running from the parent directory.
    
    Changing "export_last" to False will allow you to export from specific runs.

    More information and comments can be found in the Jupyter Notebook:
    ../notebooks/results.ipynb
    """
    path = lambda x: str(Path(x))
    export_last = True

    # 10 run ids (trained with 10 different seeds)
    date = "2023-09-22"
    ids = ["0307eblk", "162pncmd", "45dljtwk", "5fs1oqli", "6wsxi1hf",
        "7sdm9qvf", "h444g60r", "ntlx23x5", "tziecklt", "xetzm1dg"]
    # Seed 42 id: n00r57nc
    if export_last: ids = []

    cfg.wandb.mode = "disabled"
    cfg.common.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.dataset.save_format = "pkl"
    trainer = Trainer(cfg)

    for id in ids:
        ### XXX: Paths
        if export_last:
            # Checkpoint path
            checkpoint_path = path("checkpoints/last_general_checkpoint.pth")
            # Paths to save our pkl files
            counts_path = path("data/num_examples_clusters.pkl")
            counts_inter_path = path("data/num_cluster_intersections.pkl")
            # Paths to save our root files
            out_root_filename_cluster_inter = path("data/cluster_inter_nf1rate.root")
            out_root_filename = path("data/pred_nf1rate.root")
        else:
            # Checkpoint path
            checkpoint_path = path(f"checkpoints/{date}/run_{id}/general_checkpoint.pth")
            # Paths to save our pkl files
            counts_path = path(f"data/num_examples_clusters_{id}.pkl")
            counts_inter_path = path(f"data/num_cluster_intersections_{id}.pkl")
            # Paths to save our root files
            out_root_filename_cluster_inter = path(f"data/cluster_inter_nf1rate_{id}.root")
            out_root_filename = path(f"data/pred_nf1rate_{id}.root")
        
        ### XXX: Loading checkpoint
        trainer.load_checkpoints(checkpoint_path)

        trainer.model.eval()
        torch.set_grad_enabled(False)

        # Init trainer with a dataset that doesn't filter out the
        # GRBs -> we only use it to obtain the dataset with GRBs
        # May take some time as it recreates the dataset
        cfg.dataset.save_format = None
        cfg.dataset.filter_conditions = ["rate[0]/rate_err[0] > 20"]
        trainer_with_GRBs = Trainer(cfg)

        dataset_full_GRBs = trainer_with_GRBs.dataset_full
        # Fix the dataset transform to match the transform we used when training the model
        dataset_full_GRBs.transform = trainer.dataset_full.transform

        ### XXX: Prediction on full dataset with GRBs (e.g rate[0])
        # Need to transform before inputting the whole set into the model
        X = dataset_full_GRBs.X_cpu
        dataset_tensor = trainer.dataset_full.transform(X).to(device=trainer.device)

        # Apply the model trained without GRBs to the whole dataset
        # including GRBs.
        pred = trainer.model(dataset_tensor).detach().to("cpu")

        # Remove unused tensors that are on GPU
        delete(dataset_tensor)
        
        ### XXX: Some useful variables
        # Create a PyTorch Subset with all data
        data_df = dataset_full_GRBs.data_df
        subset_dataset_full_GRBs = Subset(dataset_full_GRBs,
                                    indices=range(dataset_full_GRBs.n_examples))
        # Note: We do it because our functions require PyTorch Subsets as input
        target_id_dict = trainer.dataset_full.target_names2id
        feature_id_dict = trainer.dataset_full.feature_names2id

        # Track the err rate names, they will be used to retrieve the corresponding err rates
        rate_err_names = [f'rate_err[{re.findall("[0-9]+", target_name)[0]}]'\
                        for target_name in cfg.dataset.target_names]
        
        rate_errs = get_columns(subset_dataset_full_GRBs, rate_err_names)
        # Residuals: Target - prediction
        sorted_time, sorted_y, sorted_y_hat = get_all_time_y_y_hat(subset_dataset_full_GRBs, pred)
        residuals = sorted_y-sorted_y_hat

        # Pulls
        var, var_name = residuals/rate_errs, "pull"

        # Modified gaussian fit
        new_mean, new_std = list(zip(*[find_moments(var[:, j]) for j in range(var.shape[1])]))
        new_mean, new_std = np.array(new_mean), np.array(new_std)
        
        ### XXX: Number of +/- ve examples/clusters
        k = 5
        names = ["new_std", "# +ve examples", "# -ve examples"]
        counts = pd.DataFrame([new_std, np.sum(var > k*new_std, axis=0), np.sum(var < -k*new_std, axis=0)],
                columns=cfg.dataset.target_names,
                index=names).T.astype({names[1]: 'int',
                                        names[2]: 'int'})
        # Get number of clusters too:
        cluster_counts = {f"# {(1-pred_below)*'-'+pred_below*'+'}ve clusters":
                    [get_clusters(data_df, var, new_std,
                                    target_name, target_id_dict[target_name],
                                    k=5,
                                    pred_below=pred_below,
                                    discard_w=0)[3].size for target_name in cfg.dataset.target_names]
                    for pred_below in [1, 0]}
        cluster_counts = pd.DataFrame(cluster_counts, index=cfg.dataset.target_names)
        counts = pd.concat([counts, cluster_counts], axis=1)
        counts.to_pickle(counts_path)

        print(f"Pickled counts at {counts_path}")

        # XXX: Number of +/- ve cluster intersections
        dict_counts_inter = {0: {}, 1: {}}
        dict_starts_ends = {0: {}, 1: {}}

        k, discard_w = 5, 30
        dfs = [get_merged_df(cfg, data_df, var, new_std,
                            k, pred_below, discard_w) for pred_below in [0, 1]]
        
        for pred_below, df in enumerate(dfs):
            inter_id_or_conds = list(df["inter_id"].unique())+\
                [f"# inter > {i}" for i in range(len(cfg.dataset.target_names))]

            for inter_id_or_cond in inter_id_or_conds:
                # XXX: k, pred_below, discard_w are fixed. They were
                # defined earlier
                out = get_inter_clusters(data_df, df, inter_id_or_cond)
                m_points, data, times, starts, ends, groups, integrals, lengths = out
                
                dict_counts_inter[pred_below] |= {inter_id_or_cond: starts.size}
                # For the vstack: First col: starts, Second row: ends
                dict_starts_ends[pred_below] |= {inter_id_or_cond.replace(',','_').replace(' ', '_').replace('>','more'):
                                                np.vstack([starts, ends])}
        counts_inter = pd.DataFrame(dict_counts_inter).astype(pd.Int64Dtype()).rename(columns={0: 'negative', 1: 'positive'})
        counts_inter.to_pickle(counts_inter_path)

        print(f"Pickled counts_inter at {counts_inter_path}")

        ### XXX: Export our cluster intersections in .root (ROOT CERN) format
        print("Exporting our cluster intersections in .root (ROOT CERN) format...")
        with ur.recreate(out_root_filename_cluster_inter) as file:
            file["cluster_inter_nf1rate"] = dict_starts_ends[1]  # positive examples
            file["cluster_inter_nf1rate"].show()

            file["negative_cluster_inter_nf1rate"] = dict_starts_ends[0]  # negative examples
            file["negative_cluster_inter_nf1rate"].show()

        ### XXX: Export our predictions in .root (ROOT CERN) format
        print("Exporting our predictions in .root (ROOT CERN) format...")
        with ur.recreate(out_root_filename) as file:
            my_dict = {"unix_time": data_df["unix_time"].values}
            my_dict |= {"pred_rate": pred.numpy()}
            file["pred_nf1rate"] = my_dict
            file["pred_nf1rate"].show()

if __name__ == "__main__":
    main()

    # Code to get our LaTex tables (nearly, then need to replace some characters)
    ids = ["0307eblk", "162pncmd", "45dljtwk", "5fs1oqli", "6wsxi1hf",
        "7sdm9qvf", "h444g60r", "ntlx23x5", "tziecklt", "xetzm1dg"]
    combined = pd.concat({id: pd.read_pickle(f"data/num_cluster_intersections_{id}.pkl") for id in ids})
    print((("$"+\
            combined.groupby(level=1).mean().round(decimals=3).astype(str) +\
            "\pm"+\
            combined.groupby(level=1).std().round(decimals=3).astype(str)+\
            "$").iloc[:22, :]).to_latex())
    print((("$"+\
            combined.groupby(level=1).mean().round(decimals=3).astype(str) +\
            "\pm"+\
            combined.groupby(level=1).std().round(decimals=3).astype(str)+\
            "$").iloc[22:, :]).to_latex())
    print((("$"+\
            combined.groupby(level=1).mean().round(decimals=3).astype(str) +\
            "\pm"+\
            combined.groupby(level=1).std().round(decimals=3).astype(str)+\
            "$").iloc[44:, :]).to_latex())
    
    print("\n\nnum_examples_clusters:")
    combined = pd.concat({id: pd.read_pickle(f"data/num_examples_clusters_{id}.pkl") for id in ids})
    print((("$"+\
            combined.groupby(level=1).mean().round(decimals=3).astype(str) +\
            "\pm"+\
            combined.groupby(level=1).std().round(decimals=3).astype(str)+\
            "$")).to_latex())
