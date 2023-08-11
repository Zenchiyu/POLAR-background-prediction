import gc
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
import torch

from matplotlib.scale import FuncScale
from omegaconf import DictConfig
from pathlib import Path
from trainer import Trainer
from utils import merge_torch_subsets


def get_time_y(dataset_val,
               target_name="rate[0]",
               return_argsort=False):
    dataset = dataset_val.dataset
    df = dataset.data_df
    # Column indices
    idx = dataset.column_names2id

    idx_target_name = idx[target_name]
    # Subset of dataset but could be in wrong order 
    val = dataset.data_cpu[dataset_val.indices]
    time_val = df.loc[df.index[dataset_val.indices], "unix_time"].values
    y_val = val[:, idx_target_name]
    
    # Sort by ascending time
    argsort = np.argsort(time_val)
    
    if not(return_argsort):
        return time_val[argsort], y_val[argsort]
    return argsort, time_val[argsort], y_val[argsort]

def get_time_y_y_hat(dataset_val,
                     pred,
                     target_name="rate[0]"):
    dataset = dataset_val.dataset
    tmp = get_time_y(dataset_val, target_name=target_name, return_argsort=True)
    argsort, sorted_time_val, sorted_y_val = tmp
    del tmp
    
    pred = pred.detach().cpu()

    sorted_y_hat_val = pred[:, dataset.target_names2id[target_name]][argsort]
    return sorted_time_val, sorted_y_val, sorted_y_hat_val

def get_all_time_y_y_hat(dataset_val, pred):
    dataset = dataset_val.dataset
    df = dataset.data_df
    # Column indices
    idx = dataset.column_names2id

    idx_target_names = [idx[t] for t in dataset.target_names]

    # Subset of dataset but could be in wrong order 
    val = dataset.data_cpu[dataset_val.indices]
    time_val = df.loc[df.index[dataset_val.indices], "unix_time"].values
    y_val = val[:, idx_target_names]
    pred = pred.detach().cpu()
    
    # Sort by ascending time
    argsort = np.argsort(time_val)
    return time_val[argsort], y_val[argsort], pred[argsort]

def get_columns(dataset_subset, column_names):
    dataset = dataset_subset.dataset
    df = dataset.data_df
    # Column indices
    idx = dataset.column_names2id

    idx_column_names = [idx[c] for c in column_names]

    # Subset of dataset but could be in wrong order
    subset = dataset.data_cpu[dataset_subset.indices]
    time = df.loc[df.index[dataset_subset.indices], "unix_time"].values
    columns = subset[:, idx_column_names]

    # Sort by ascending time
    argsort = np.argsort(time)
    return columns[argsort]

def get_column(dataset_subset, column_name):
    return get_columns(dataset_subset, [column_name])

def find_moments(data, verbose=False):
    """
    Find moments, especially std, for gaussian fit
    'ignoring' outliers.
    """
    low = -np.inf
    high = np.inf
    prev_std = np.inf
    std = np.std(data)
    mean = np.mean(data)
    
    while ~np.isclose(prev_std, std):
        # Update interval
        low = -3*std + mean
        high = 3*std + mean
        
        prev_std = std
        std = np.std(data[(data>low) & (data<high)])
        if verbose: print(mean, std, low, high)
    return mean, std

def plot_normalized_hist(data,
                         mean,
                         std,
                         transform="sqrt",
                         title="Normalized Histogram",
                         xlabel="Residuals",
                         save_path="results/images/residual_hist.png"):
    """
    Plot normalized histogram along with a gaussian specified by mean, std.
    The histogram can be visualized with different yscale specified by 'transform',
    e.g sqrt transform.
    """

    # Gaussian
    f = lambda x, mean, std: 1/np.sqrt(2*np.pi*std**2)*np.exp(-(x-mean)**2/(2*std**2))
    
    fig, ax = plt.subplots()
    ax.hist(data, bins=500, alpha=0.5, density=True)

    xs = np.linspace(data.min(), data.max(), 255)
    ax.plot(xs, f(xs, mean, std), zorder=np.inf,
            color="m", linewidth=1, linestyle="--")
    ax.plot([-5*std, -5*std], [0, f(xs, mean, std).max()/36],
            'r', label=r"$-5\sigma$")
    ax.plot([5*std, 5*std], [0, f(xs, mean, std).max()/36],
            'g', label=r"$+5\sigma$")
    ax.set_xlabel(xlabel)
    ax.legend()
    if transform == "sqrt":
        ax.set_yscale(FuncScale(0, (lambda x: np.sqrt(x),
                                    lambda x: np.power(x, 2))))
        if title is not None: ax.set_title("Sqrt Normalized Histogram")
    else:
        ax.set_title(title)
    if save_path: plt.savefig(save_path)

    return fig

#############################################################################
def plot_val_target_against_time(dataset_val,
                                 target_name="rate[0]",
                                 save_path="results/images/target.png"):
    """
    Plot one measurement "target_name" wrt unix_time
    for the validation set
    """
    sorted_time_val, sorted_y_val = get_time_y(dataset_val, target_name)
    
    _, ax = plt.subplots()
    ax.plot(sorted_time_val, sorted_y_val, '-g', linewidth=0.1)
    ax.set_xlabel("Tunix [s]")
    ax.set_ylabel(f"{target_name}")
    ax.set_title(f"{target_name} wrt unix time for validation seet")
    if save_path: plt.savefig(save_path)

def plot_loss(train_loss,
              val_loss,
              epoch=None,
              save_path="results/images/loss.png"):
    plt.figure()
    if epoch:
        plt.plot(train_loss[:epoch+1], label="train")
        plt.plot(val_loss[:epoch+1], label="val")
    else:
        plt.plot(train_loss, label="train")
        plt.plot(val_loss, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    if save_path: plt.savefig(save_path)

def plot_val_prediction_target(dataset_val,
                               pred,
                               target_name="rate[0]",
                               save_path="results/images/pred_target.png"):  # TODO: use a better name
    tmp = get_time_y_y_hat(dataset_val, pred, target_name)
    sorted_time_val, sorted_y_val, sorted_y_hat_val = tmp
    del tmp

    _, ax = plt.subplots()
    ax.plot(sorted_time_val, sorted_y_val,
            '-g', linewidth=0.1, label="true")
    ax.plot(sorted_time_val, sorted_y_hat_val,
            '-r', linewidth=0.1, label="predicted")
    ax.set_xlabel("Tunix [s]")
    ax.set_ylabel(f"{target_name.capitalize()}")  # Nb. photons per second: [Hz] if rate[i]
    ax.set_title(f"Prediction of {target_name.capitalize()}")
    if save_path: plt.savefig(save_path)

def plot_val_residual(dataset_val,
                      pred,
                      target_name="rate[0]",
                      save_path="results/images/residual_plot.png",
                      save_path_hist="results/images/residual_hist.png"):
    tmp = get_time_y_y_hat(dataset_val, pred, target_name)
    sorted_time_val, sorted_y_val, sorted_y_hat_val = tmp
    del tmp


    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.03
    rect_residuals = [left, bottom, width, height]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_axes(rect_residuals)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.set_title("Normalized Histogram (Density)")

    residuals = sorted_y_val-sorted_y_hat_val
    print("res shape", residuals.shape)
    new_mean, new_std = find_moments(residuals)
    
    # Residuals
    ax.plot(sorted_time_val, residuals, '-r', linewidth=0.1)
    ax.set_xlabel("Tunix [s]")
    ax.set_ylabel(f"{target_name}")
    ax.set_title(f"Residual plot of {target_name}")
    
    # Gaussian fit
    xs = np.linspace(residuals.min(), residuals.max(), 255)
    f = lambda x, mean, std: 1/np.sqrt(2*np.pi*std**2)*np.exp(-(x-mean)**2/(2*std**2))
    ax_histy.plot(f(xs, new_mean, new_std), xs, zorder=np.inf, color="m", linewidth=1, linestyle="--")
    # Histogram residuals
    _ = sns.histplot(data=residuals.to_frame(target_name),
                     y=target_name,
                     stat="density",
                     ax=ax_histy)
    if save_path: plt.savefig(save_path)

    ######
    
    plot_normalized_hist(residuals,
                         new_mean,
                         new_std,
                         save_path=save_path_hist)
    
def plot_val_pull(dataset_val,
                      pred,
                      target_name="rate[0]",
                      rate_err_name="rate_err[0]",
                      save_path="results/images/pull_plot.png",
                      save_path_hist="results/images/pull_hist.png"):
    # TODO: is it really a pull ? to do residual/rate_err
    tmp = get_time_y_y_hat(dataset_val, pred, target_name)
    sorted_time_val, sorted_y_val, sorted_y_hat_val = tmp
    del tmp


    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.03
    rect_pulls = [left, bottom, width, height]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_axes(rect_pulls)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.set_title("Normalized Histogram (Density)")

    residuals = sorted_y_val-sorted_y_hat_val
    rate_err = get_column(dataset_val, rate_err_name)
    print("res shape", residuals.shape)
    print("rate_err shape", rate_err.shape)
    

    pulls = residuals/rate_err
    new_mean, new_std = find_moments(pulls)
    
    # Pulls
    ax.plot(sorted_time_val, pulls, '-r', linewidth=0.1)
    ax.set_xlabel("Tunix [s]")
    ax.set_ylabel(f"{target_name}")
    ax.set_title(f"Pull plot of {target_name}")
    
    # Gaussian fit
    xs = np.linspace(pulls.min(), pulls.max(), 255)
    f = lambda x, mean, std: 1/np.sqrt(2*np.pi*std**2)*np.exp(-(x-mean)**2/(2*std**2))
    ax_histy.plot(f(xs, new_mean, new_std), xs, zorder=np.inf, color="m", linewidth=1, linestyle="--")
    # Histogram pull
    _ = sns.histplot(data=pulls.to_frame(target_name),
                     y=target_name,
                     stat="density",
                     ax=ax_histy)
    if save_path: plt.savefig(save_path)

    ######
    
    plot_normalized_hist(pulls,
                         new_mean,
                         new_std,
                         xlabel=f"Residuals/{rate_err_name}",
                         save_path=save_path_hist)

def plot_prediction_target_zoom(dataset,
                               pred,
                               target_name="rate[0]",
                               save_path="results/images/pred_target_zoom.png"):  # TODO: use a better name
    tmp = get_time_y_y_hat(dataset, pred, target_name)
    sorted_time, sorted_y, sorted_y_hat = tmp
    del tmp

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    for i, (low_n, high_n) in enumerate([(1000, 1200), (2000, 2200),
                                    (10000, 10200), (12000, 12200)]):
        axs[i//2, i%2].plot(sorted_time[low_n:high_n], sorted_y[low_n:high_n], '-g', linewidth=0.5)
        axs[i//2, i%2].plot(sorted_time[low_n:high_n], sorted_y_hat[low_n:high_n], '-r', linewidth=0.5)
        axs[i//2, i%2].set_xlabel("Tunix [s]")
        axs[i//2, i%2].set_ylabel(f"{target_name.capitalize()}")  # Nb. photons per second: [Hz] if rate[i]
        axs[i//2, i%2].set_title(f"{target_name.capitalize()}: l={low_n}, h={high_n}")
    plt.tight_layout()
    if save_path: plt.savefig(save_path)


def plot_train_val_prediction_target_zoom(trainer,
                                          dataset_train_val,
                                          pred_train_val,
                                          target_name="rate[0]",
                                          save_path="results/images/pred_target_zoom_train_val.png"):  # TODO: use a better name
    # train + val
    tmp = get_time_y_y_hat(dataset_train_val, pred_train_val, target_name)
    sorted_time, sorted_y, sorted_y_hat = tmp
    del tmp

    # which part of train + val is the training set:
    mask_train = np.isin(dataset_train_val.indices,
                         trainer.dataset_train.indices)

    # val
    tmp = get_time_y(trainer.dataset_val, target_name)
    sorted_time_val, sorted_y_val = tmp
    del tmp

    # mask_val = np.zeros(dataset_full.n_examples)
    # mask_val[dataset_full.dataset_val.indices] = 1
    

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    for i, (low_n, high_n) in enumerate([(1000, 1200), (2000, 2200),
                                    (10000, 10200), (12000, 12200)]):
        # low_n and high_n: related to validation set, not train + val
        time_val = sorted_time_val[low_n:high_n]
        min_time, max_time = time_val.min(), time_val.max()
        mask_time = np.array(((sorted_time >= min_time) & (sorted_time <= max_time)))
        
        # Train
        axs[i//2, i%2].plot(sorted_time[mask_time & mask_train],
                            sorted_y[mask_time & mask_train], '.-c', linewidth=0.5)
        # Validation
        axs[i//2, i%2].plot(time_val, sorted_y_val[low_n:high_n], '.-g', linewidth=0.5)
        
        # Prediction using train + val
        axs[i//2, i%2].plot(sorted_time[mask_time],
                            sorted_y_hat[mask_time], '-r', linewidth=0.5)
        axs[i//2, i%2].set_xlabel("Tunix [s]")
        axs[i//2, i%2].set_ylabel(f"{target_name.capitalize()}")  # Nb. photons per second: [Hz] if rate[i]
        axs[i//2, i%2].set_title(f"{target_name.capitalize()}: l={low_n}, h={high_n}")
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    
def get_start_length_intensity_clusters():
    pass

@hydra.main(version_base=None, config_path="../config", config_name="trainer")
def main(cfg: DictConfig):
    ## Don't start a wandb run
    cfg.wandb.mode = "disabled"
    
    
    cfg.dataset.save_format = "pkl"
    # Comment prev. line and uncomment this below
    # once we're sure that we don't change anymore the dataset:
    
    ## Save dataset or load it
    # p = Path(cfg.dataset.filename)
    # filename =  f"{str(p.parent)}/{p.stem}_dataset.pkl"
    # if Path(filename).is_file():  # if exists and is a file
    #     cfg.dataset.filename = filename
    # else:
    #     cfg.dataset.save_format = "pkl"  # to save dataset
    
    trainer = Trainer(cfg)
    
    ### Loading checkpoint
    general_checkpoint = torch.load("checkpoints/last_general_checkpoint.pth")
    trainer.model.load_state_dict(general_checkpoint["model_state_dict"])
    trainer.optimizer.load_state_dict(general_checkpoint["optimizer_state_dict"])

    trainer.epoch = general_checkpoint["epoch"]
    trainer.train_loss = general_checkpoint["train_loss"].cpu()
    trainer.val_loss = general_checkpoint["val_loss"].cpu()
    
    del general_checkpoint

    ### Plotting
    trainer.model.eval()
    with torch.no_grad():
        # https://discuss.pytorch.org/t/how-to-delete-a-tensor-in-gpu-to-free-up-memory/48879/15
        ## targets wrt unix_time for validation set
        # for i, target_name in enumerate(cfg.dataset.target_names):
        #     plot_val_target_against_time(trainer.dataset_val,
        #                                  target_name=target_name,
        #                                  save_path=f"results/images/target_{i}.png")
        
        # ## Loss
        # plot_loss(trainer.train_loss, trainer.val_loss)
        
        ## Prediction on validation set (e.g rate[0])
        # Need to transform before inputting the whole validation set into
        # the model
        dataset_full = trainer.dataset_full
        val_tensor = dataset_full.X_cpu[trainer.dataset_val.indices]
        val_tensor = dataset_full.transform(val_tensor).to(cfg.common.device)
        pred = trainer.model(val_tensor)
        
        for i, target_name in enumerate(cfg.dataset.target_names):
            plot_val_prediction_target(trainer.dataset_val,
                                       pred,
                                       target_name=target_name,
                                       save_path=f"results/images/pred_target_{i}.png")
            # Closer look/zoomed in for some regions
            plot_prediction_target_zoom(trainer.dataset_val,
                                        pred,
                                        target_name=target_name,
                                        save_path=f"results/images/pred_target_zoom_{i}.png")
            
        ## Residuals + hist + gaussian fit
        for i, target_name in enumerate(cfg.dataset.target_names):
            if target_name not in [f"rate[{i}]" for i in range(13)]:
                plot_val_residual(trainer.dataset_val,
                                  pred,
                                  target_name=target_name,
                                  save_path=f"results/images/residual_plot_{i}.png",
                                  save_path_hist=f"results/images/residual_hist_{i}.png")
            else:
                j = re.findall("[0-9]+", target_name)[0]
                plot_val_pull(trainer.dataset_val,
                              pred,
                              target_name=target_name,
                              rate_err_name=f"rate_err[{j}]",
                              save_path=f"results/images/pull_plot_{i}.png",
                              save_path_hist=f"results/images/pull_hist_{i}.png")
        
        ## Prediction on both train + val set
        pred = pred.to(device="cpu")
        val_tensor = val_tensor.to(device="cpu")
        
        del pred
        del val_tensor
        torch.cuda.empty_cache()
        gc.collect()

        dataset_train_val = merge_torch_subsets([trainer.dataset_train,
                                                trainer.dataset_val])
        train_val_tensor = dataset_full.X_cpu[dataset_train_val.indices]
        train_val_tensor = dataset_full.transform(train_val_tensor).to(cfg.common.device)

        pred_train_val = trainer.model(train_val_tensor)
        print("after pred_train_val", print(torch.cuda.memory_allocated(device="cuda")))

        for i, target_name in enumerate(cfg.dataset.target_names):
            plot_train_val_prediction_target_zoom(trainer,
                                                dataset_train_val,
                                                pred_train_val,
                                                target_name=target_name,
                                                save_path=f"results/images/pred_target_zoom_train_val_{i}.png")
            
        ## Comment this line below if don't want to show
        plt.show()


if __name__ == "__main__":
     main()
