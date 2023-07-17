import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from dataset import PolarDataset
from torch.utils.data import DataLoader, random_split
from torch import nn
from torch.optim import Adam
from tqdm import tqdm


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        print(OmegaConf.to_yaml(cfg))
        
        
        self.run = wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
                              **cfg.wandb)
        
        self.cfg = cfg
        self.seed = self.cfg.common.seed
        torch.manual_seed(self.seed)
        
        self.n_epochs = self.cfg.common.n_epochs
        
        if self.cfg.common.device is not None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu" 
        else:
            self.device = self.cfg.common.device
        
        ### Dataset
        self.dataset_full = PolarDataset(cfg.dataset.filename,
                                    cfg.dataset.feature_names,
                                    cfg.dataset.target_names,
                                    self.device)
        
        # Split train, validation, test
        dataset_train, dataset_val, dataset_test = random_split(self.dataset_full,
                                                                  [cfg.dataset.train.size,
                                                                   cfg.dataset.val.size,
                                                                   cfg.dataset.test.size])
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        # TODO: use something else instead of random split
        
        # Process data
        # TODO: apply centering and reducing
        
        # Dataloaders
        self.train_loader = DataLoader(dataset_train,
                                  batch_size=cfg.dataset.train.batch_size,
                                  shuffle=cfg.dataset.train.shuffle)
        self.val_loader = DataLoader(dataset_val,
                                  batch_size=cfg.dataset.val.batch_size)
        self.test_loader = DataLoader(dataset_test,
                                  batch_size=cfg.dataset.test.batch_size)
        
        
        ### Model
        print(self.dataset_full.n_features, self.dataset_full.n_targets)
        # TODO: place the hyperparams in a config
        model = nn.Sequential(
            nn.Linear(self.dataset_full.n_features, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, self.dataset_full.n_targets),
            )
        
        ### Criterion
        criterion = nn.MSELoss()
        
        ### Optimizer
        self.optimizer = Adam(model.parameters(), **cfg.optimizer.hyperparams)
        
        ### Move to device (e.g "cuda"). Data was already on device
        self.model = model.to(device=self.device)
        self.criterion = criterion.to(device=self.device)
        
    
    def fit(self) -> None:
        ## Metrics
        train_loss = []
        val_loss = []
        
        for e in tqdm(range(self.n_epochs)):
            self.model.train()
            train_epoch_loss = 0
            for batch in self.train_loader:
                x, y = batch
                y_hat = self.model(x)
                
                loss = self.criterion(y_hat, y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                train_epoch_loss += loss.item()
                
            train_loss.append(train_epoch_loss)
            
            ## Validation set
            self.model.eval()
            val_epoch_loss = 0
            for batch in self.val_loader:
                x, y = batch
                y_hat = self.model(x)
                
                loss = self.criterion(y_hat, y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                val_epoch_loss += loss.item()
                
            val_loss.append(val_epoch_loss)
            
            self.model.eval()
            self.run.log({"epoch": e,
                          "train/loss": train_epoch_loss,
                          "val/loss": val_epoch_loss})
            
        self.run.finish()
        return train_loss, val_loss
