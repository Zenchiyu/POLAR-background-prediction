import hydra
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer  

@hydra.main(version_base=None, config_path="../config", config_name="trainer")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    # train_loss, val_loss = trainer.fit()
    # return trainer, train_loss, val_loss
    trainer.fit()

if __name__ == "__main__":
     # trainer, train_loss, val_loss = main()
     main()