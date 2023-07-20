import hydra
import torch
import typing
import wandb

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Lambda
from typing import Optional

from dataset import PolarDataset
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        print("Config:")
        print(OmegaConf.to_yaml(cfg))
        
        self.run = wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
                              **cfg.wandb)
        
        self.cfg = cfg
        self.seed = self.cfg.common.seed
        torch.manual_seed(self.seed)
        
        self.n_epochs = self.cfg.common.n_epochs
        
        if self.cfg.common.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu" 
        else:
            self.device = self.cfg.common.device
        
        print(f"Using device: {self.device}")
        
        ### Datasets: train
        self.init_datasets()
        
        ### Model
        # TODO: place the hyperparams in a config
        model = nn.Sequential(
            nn.Linear(self.dataset_full.n_features, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
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
        
        ### Logging more info into wandb (e.g model architecture with log_graph)
        if self.cfg.wandb_watch:
            self.run.watch(self.model, self.criterion,
                           log="all", log_graph=True)
    
    def fit(self) -> None:
        """
        Train the model using the training set
        """
        ## Metrics
        train_loss = torch.zeros(self.n_epochs)
        val_loss = torch.zeros(self.n_epochs)
        
        # TODO: add second end_condition related to stagnation of training loss
        for epoch in tqdm(range(self.n_epochs)):
            ## Updating model with one pass through training set
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
                
            train_loss[epoch] = train_epoch_loss/len(self.dataset_train)
            
            ## Validation set, evaluation using current model
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
                
            val_loss[epoch] = val_epoch_loss/len(self.dataset_val)
            
            # Wandb log
            self.run.log({"epoch": epoch,
                          "train/loss": train_loss[epoch],
                          "val/loss": val_loss[epoch]})
            
            # Save checkpoints
            torch.save(self.model.state_dict(), "checkpoints/model.pth")
            torch.save(self.optimizer.state_dict(), "checkpoints/optimizer.pth")
            torch.save(epoch, "checkpoints/epoch.pth")
            torch.save(train_loss, "checkpoints/train_loss.pth")
            torch.save(val_loss, "checkpoints/val_loss.pth")
        
        # Wandb finish
        self.run.finish()
        
        
    def init_datasets(self) -> tuple[DataLoader, ...]:
        """
        Create:
        - self.dataset_full (Pytorch Dataset) with:
            - self.dataset_full.transform: (x-mean_train)/std_train
            
        - self.dataset_train, self.dataset_val, self.dataset_test (Pytorch Subset)
        - self.train_loader, self.val_loader, self.test_loader (Pytorch DataLoader)
        
        Links:
        - https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
        - https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset
        - https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        """
        self.dataset_full = PolarDataset(self.cfg.dataset.filename,
                                    self.cfg.dataset.feature_names,
                                    self.cfg.dataset.target_names,
                                    self.device,
                                    new_columns=self.cfg.dataset.new_columns,
                                    save_format=self.cfg.dataset.save_format)
        
        ### Split train, validation, test
        split_percentages = [self.cfg.dataset.train.size,
                             self.cfg.dataset.val.size,
                             self.cfg.dataset.test.size]
        # TODO: use something else instead of random split
        datasets = random_split(self.dataset_full, split_percentages)
        self.dataset_train, self.dataset_val, self.dataset_test = datasets
        
        ### Process features by applying centering and reducing
        data_train_tensor = self.dataset_train.dataset.X[self.dataset_train.indices]
        mean_train = data_train_tensor.mean(dim=0)
        std_train = data_train_tensor.std(dim=0)
        
        self.dataset_full.transform =  Lambda(lambda x: (x-mean_train)/std_train)
        
        # XXX: be careful when evaluating on train or test set, we need to be sure
        # that we're using the same transform !
        
        ### Dataloaders
        self.train_loader = DataLoader(self.dataset_train,
                                  batch_size=self.cfg.dataset.train.batch_size,
                                  shuffle=self.cfg.dataset.train.shuffle)
        self.val_loader = DataLoader(self.dataset_val,
                                  batch_size=self.cfg.dataset.val.batch_size)
        self.test_loader = DataLoader(self.dataset_test,
                                  batch_size=self.cfg.dataset.test.batch_size)
        
        return self.train_loader, self.val_loader, self.test_loader