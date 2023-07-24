import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from matplotlib.scale import FuncScale
from omegaconf import DictConfig
from pathlib import Path
from trainer import Trainer


def get_time_y(dataset_val,
               target_name="rate[0]"):
    dataset = dataset_val.dataset

    # Pandas dataframes
    X = dataset.data_df[dataset.feature_names]
    y = dataset.data_df[dataset.target_names]
    
    X_val = X.iloc[dataset_val.indices]
    y_val = y.iloc[dataset_val.indices]
    
    X_val.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)

    print(X_val.columns)
    argsort = np.argsort(X_val["unix_time"])[::-1]
    sorted_time_val = X_val.loc[argsort, "unix_time"]
    sorted_y_val = y_val.loc[argsort, target_name]

    return sorted_time_val, sorted_y_val

def get_time_y_y_hat(dataset_val,
                     pred,
                     target_name="rate[0]"):
    dataset = dataset_val.dataset
    target_names = dataset.target_names
    idx_target_name = target_names.index(target_name)

    pred = pred.cpu().detach().numpy()

    # Pandas dataframes
    X = dataset.data_df[dataset.feature_names]
    y = dataset.data_df[dataset.target_names]
    
    X_val = X.iloc[dataset_val.indices]
    y_val = y.iloc[dataset_val.indices]
    
    X_val.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)

    print(X_val.columns)
    argsort = np.argsort(X_val["unix_time"])[::-1]
    sorted_time_val = X_val.loc[argsort, "unix_time"]
    sorted_y_val = y_val.loc[argsort, target_name]
    sorted_y_hat_val = pred[argsort, idx_target_name]

    return sorted_time_val, sorted_y_val, sorted_y_hat_val

def find_moments(data, verbose=False):
    """
    Find moments, especially std, for gaussian fit
    'ignoring' outliers
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
    ax.set_xlabel("Residuals")
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
        plt.plot(train_loss[:epoch+1])
        plt.plot(val_loss[:epoch+1])
    else:
        plt.plot(train_loss)
        plt.plot(val_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if save_path: plt.savefig(save_path)

def plot_val_prediction_target(dataset_val,
                               pred,
                               target_name="rate[0]",
                               save_path=f"results/images/pred_target.png"):  # TODO: use a better name
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
    new_mean, new_std = find_moments(residuals)
    
    # Residuals
    ax.plot(sorted_time_val, residuals, '-r', linewidth=0.1)
    ax.set_xlabel("Tunix [s]")
    ax.set_ylabel(f"{target_name}")  # Nb. photons per second (averaged over each bin)
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
    

@hydra.main(version_base=None, config_path="../config", config_name="trainer")
def main(cfg: DictConfig):
    # don't want to start a wandb run
    cfg.wandb.mode = "disabled"
    
    ## Save dataset or load it
    # cfg.dataset.save_format = "pkl"  # to save dataset
    p = Path(cfg.dataset.filename)
    cfg.dataset.filename = f"{str(p.parent)}/{p.stem}_dataset.pkl"
    # Comment the previous lines if don't want to save dataset or load it

    trainer = Trainer(cfg)
    
    ### Loading checkpoint
    general_checkpoint = torch.load("checkpoints/last_general_checkpoint.pth")
    trainer.model.load_state_dict(general_checkpoint["model_state_dict"])
    trainer.optimizer.load_state_dict(general_checkpoint["optimizer_state_dict"])

    trainer.epoch = general_checkpoint["epoch"]
    torch.train_loss = general_checkpoint["train_loss"]
    torch.val_loss = general_checkpoint["val_loss"]
    
    trainer.model.eval()
    
    ### Plotting
    ## targets wrt unix_time for validation set
    for target_name in cfg.dataset.target_names:
        plot_val_target_against_time(trainer.dataset_val,
                                 target_name=target_name)
    
    ## Loss
    plot_loss(torch.train_loss, torch.val_loss)
    
    ## Prediction on validation set (e.g rate[0])
    # Need to transform before inputting the whole validation set into
    # the model
    dataset_full = trainer.dataset_full
    dataset_val_tensor = trainer.dataset_val.dataset.X[trainer.dataset_val.indices]
    dataset_val_tensor = dataset_full.transform(dataset_val_tensor)

    pred = trainer.model(dataset_val_tensor)
    for target_name in cfg.dataset.target_names:
        plot_val_prediction_target(trainer.dataset_val,
                                   pred,
                                   target_name=target_name)
    
    ### Residuals + hist + gaussian fit
    for target_name in cfg.dataset.target_names:
        plot_val_residual(trainer.dataset_val,
                                   pred,
                                   target_name=target_name)
    
    ### Comment this line below if don't want to show
    plt.show()


if __name__ == "__main__":
     main()