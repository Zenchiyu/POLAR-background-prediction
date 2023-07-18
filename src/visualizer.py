import torch
import numpy as np
import pandas as pd
import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer  



def plot_loss(train_loss, val_loss, epoch=None):
    plt.figure()
    if epoch:
        plt.plot(train_loss[:epoch+1])
        plt.plot(val_loss[:epoch+1])
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
    ax.set_xlabel("Quantized Tunix [s]")
    ax.set_ylabel("Avg Rate[0] [Hz]")  # Nb. photons per second (averaged over each bin)
    ax.set_title("Light curve of Rate[0]")
    plt.savefig("results/images/pred_rate_0.png")



@hydra.main(version_base=None, config_path="../config", config_name="trainer")
def main(cfg: DictConfig):
    cfg.wandb.mode = "disabled"
    trainer = Trainer(cfg)
    
    trainer.model.load_state_dict(torch.load("checkpoints/model.pth"))
    trainer.optimizer.load_state_dict(torch.load("checkpoints/optimizer.pth"))

    trainer.epoch = torch.load("checkpoints/epoch.pth")
    torch.train_loss = torch.load("checkpoints/train_loss.pth")
    torch.val_loss = torch.load("checkpoints/val_loss.pth")
    
    trainer.model.eval()
    
    
    plot_loss(torch.train_loss, torch.val_loss)
    plot_val_prediction_rate_0(trainer.dataset_val,
                               trainer.model(trainer.dataset_val.dataset.X[trainer.dataset_val.indices]))
    

if __name__ == "__main__":
     main()