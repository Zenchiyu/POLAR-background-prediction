import torch
# import wandb
from dataset import PolarDataset
from torch.utils.data import DataLoader, random_split
from torch import nn
from torch.optim import Adam

class Trainer:
    def __init__(self, cfg) -> None:
        # wandb.init(config=cfg.wandb)
        
        self.cfg = cfg
        self.seed = self.cfg.seed
        torch.manual_seed(self.seed)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 
        # torch.device(cfg.device)
        
        ### Dataset
        self.dataset_full = PolarDataset(cfg.dataset.filename,
                                    cfg.dataset.feature_names,
                                    cfg.dataset.target_names,
                                    self.device)
        
        # Split train, validation, test
        dataset_train, dataset_val, dataset_test = random_split(self.dataset_full,
                                                                  [cfg.dataset.train_size,
                                                                   cfg.dataset.val_size,
                                                                   cfg.dataset.test_size])
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        # TODO: use something else instead of random split
        
        # Process data
        # TODO: apply centering and reducing
        
        # Dataloaders
        self.train_loader = DataLoader(dataset_train,
                                  batch_size=cfg.dataset.train_batch_size,
                                  shuffle=cfg.dataset.train_shuffle)
        self.val_loader = DataLoader(dataset_val,
                                  batch_size=cfg.dataset.val_batch_size)
        self.test_loader = DataLoader(dataset_test,
                                  batch_size=cfg.dataset.test_batch_size)
        
        
        ### Model
        # TODO: place the hyperparams in a config
        model = nn.Sequential(
            nn.Linear(self.dataset_full.n_features, 100),
            nn.ReLU(),
            nn.Linear(self.dataset_full.n_features, 100),
            nn.ReLU(),
            nn.Linear(self.dataset_full.n_features, self.dataset_full.n_targets),
            )
        
        ### Criterion
        criterion = nn.MSELoss()
        
        ### Optimizer
        self.optimizer = Adam(model.parameters(), **cfg.optimizer.hyperparams)
        
        ### Move to device (e.g "cuda"). Data was already on device
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        
