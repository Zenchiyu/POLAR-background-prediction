import hydra
import torch

from omegaconf import DictConfig
from pathlib import Path
from trainer import Trainer
from utils import delete


@hydra.main(version_base=None, config_path="../config", config_name="trainer")
def main(cfg: DictConfig):
    ## Don't start a wandb run
    cfg.wandb.mode = "disabled"
    ## Use the GPU when available, otherwise use the CPU
    cfg.common.device = "cuda" if torch.cuda.is_available() else "cpu"

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
    trainer.load_checkpoints("checkpoints/last_general_checkpoint.pth")

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

    ### Prediction on full dataset with GRBs (e.g rate[0])
    # Need to transform before inputting the whole set into the model
    X = dataset_full_GRBs.X_cpu
    dataset_tensor = trainer.dataset_full.transform(X).to(device=trainer.device)

    # Apply the model trained without GRBs to the whole dataset
    # including GRBs.
    pred = trainer.model(dataset_tensor).detach().to("cpu")

    # Remove unused tensors that are on GPU
    delete(dataset_tensor)


if __name__ == "__main__":
     main()
