import hydra
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer  

@hydra.main(version_base=None, config_path="../config", config_name="trainer")
def main(cfg: DictConfig) -> None:
    trainer = Trainer(cfg)
    trainer.fit()

if __name__ == "__main__":
     main()