import hydra
import uproot as ur
import torch

from omegaconf import DictConfig
from trainer import Trainer
from utils import delete


@hydra.main(version_base=None, config_path="../config", config_name="trainer")
def main(cfg: DictConfig):
    cfg.wandb.mode = "disabled"
    cfg.common.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.dataset.save_format = "pkl"
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

    ### Export our predictions in .root (ROOT CERN) format
    print("Exporting our predictions in .root (ROOT CERN) format...")
    data_df = trainer_with_GRBs.dataset_full.data_df
    out_root_filename = 'data/pred_nf1rate.root'
    with ur.recreate(out_root_filename) as file:
        my_dict = {"unix_time": data_df["unix_time"].values}
        my_dict |= {"pred_rate": pred.numpy()}
        file["pred_nf1rate"] = my_dict
        file["pred_nf1rate"].show()

if __name__ == "__main__":
     main()
