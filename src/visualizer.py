import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from omegaconf import DictConfig, OmegaConf
from trainer import Trainer  


def plot_loss(train_loss, val_loss, epoch=None):
    plt.figure()
    if epoch:
        plt.plot(train_loss[:epoch+1])
        plt.plot(val_loss[:epoch+1])
    else:
        plt.plot(train_loss)
        plt.plot(val_loss)
    plt.savefig("results/images/loss.png")
    

def plot_val_prediction_rate_0(dataset_val, pred):
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
    sorted_time_val = X_val["unix_time"][argsort]
    sorted_y_val_r0 = y_val.loc[:, "rate[0]"][argsort]
    sorted_val_r0 = pred.cpu().detach().numpy()[:, 0][argsort]

    fig_val_prediction_rate_0, ax = plt.subplots()
    ax.plot(sorted_time_val, sorted_y_val_r0, '-g', linewidth=0.1)
    ax.plot(sorted_time_val, sorted_val_r0, '-r', linewidth=0.1)
    ax.set_xlabel("Tunix [s]")
    ax.set_ylabel("Avg Rate[0] [Hz]")  # Nb. photons per second (averaged over each bin)
    ax.set_title("Light curve of Rate[0]")
    plt.savefig("results/images/pred_rate_0.png")

def plot_val_target_against_time(dataset_val,
                                 target_name="rate[0]"):
    # Plot one measurement "target_name" wrt unix_time
    # for the validation set
    dataset = dataset_val.dataset
    data_df = dataset.data_df
    data_df_val = data_df.iloc[dataset_val.indices]
    
    data_df_val.reset_index(drop=True, inplace=True)
    data_df_val.reset_index(drop=True, inplace=True)
    
    time = data_df_val.loc[:, "unix_time"]
    argsort = np.argsort(time)[::-1]
    sorted_time_val = time[argsort]
    sorted_y_val_r0 = data_df_val.loc[:, target_name][argsort]
    
    fig_val_prediction_rate_0, ax = plt.subplots()
    ax.plot(sorted_time_val, sorted_y_val_r0, '-g', linewidth=0.1)
    ax.set_xlabel("Tunix [s]")
    ax.set_ylabel(f"{target_name}")
    ax.set_title(f"{target_name} wrt unix time for validation seet")
    plt.savefig("results/images/target.png")


@hydra.main(version_base=None, config_path="../config", config_name="trainer")
def main(cfg: DictConfig):
    # don't want to start a wandb run
    cfg.wandb.mode = "disabled"
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
    ## Loss
    plot_loss(torch.train_loss, torch.val_loss)
    
    ## Prediction on validation set (predicting rate 0)
    # Need to transform before inputting the whole validation set into
    # the model
    dataset_full = trainer.dataset_full
    dataset_val_tensor = trainer.dataset_val.dataset.X[trainer.dataset_val.indices]
    dataset_val_tensor = dataset_full.transform(dataset_val_tensor)
    plot_val_prediction_rate_0(trainer.dataset_val,
                               trainer.model(dataset_val_tensor))
    
    ## rate[0]/rate_err[0] wrt unix_time for validation set
    plot_val_target_against_time(trainer.dataset_val,
                                 target_name=["rate[0]/rate_err[0]"])
    

if __name__ == "__main__":
     main()