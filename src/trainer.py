import os
import shutil
import torch
import wandb

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Lambda
from typing import Optional

from dataset import PolarDataset
from datetime import date
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
        torch.manual_seed(self.seed)  # TODO: careful about numpy or random libraries
        
        self.n_epochs = self.cfg.common.n_epochs
        
        if self.cfg.common.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu" 
        else:
            self.device = self.cfg.common.device
        
        print(f"Using device: {self.device}")
        
        ### Datasets: train, validation, test sets
        self.init_datasets()

        ### Model
        model = self.create_model()
        
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
        Train the model using the training set.

        Note: the losses per epoch that we log are biased,
        especially train loss as the model changes in between. 

        Link(s):
        - https://stats.stackexchange.com/questions/436154/is-it-better-to-accumulate-accuracy-and-loss-during-an-epoch-or-recompute-all-of/608648#608648
        - https://stackoverflow.com/questions/54053868/how-do-i-get-a-loss-per-epoch-and-not-per-batch
        - https://stackoverflow.com/questions/55627780/evaluating-pytorch-models-with-torch-no-grad-vs-model-eval
        - https://discuss.pytorch.org/t/loss-changes-with-torch-no-grad/30806/5
        """
        self.begin_date = str(date.today())

        ## Copy config file into specific run folder for future usage.
        self.save_config()
            
        ## Metrics
        train_loss = torch.zeros(self.n_epochs, device=self.device)
        val_loss = torch.zeros(self.n_epochs, device=self.device)
        
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
                
            train_loss[epoch] = train_epoch_loss/len(self.train_loader)
            
            ## Validation set, evaluation using current model
            # TODO: add toggle whether want evaluation or not and
            # TODO: update config file where we can choose to toggle or not
            self.model.eval()
            with torch.no_grad():
                val_epoch_loss = 0
                for batch in self.val_loader:
                    x, y = batch
                    y_hat = self.model(x)
                    
                    loss = self.criterion(y_hat, y)
                    val_epoch_loss += loss.item()
                
            val_loss[epoch] = val_epoch_loss/len(self.val_loader)
            
            # Wandb log
            self.run.log({"epoch": epoch,
                          "train/loss": train_loss[epoch],
                          "val/loss": val_loss[epoch]})
            
            # Save general checkpoint
            self.save_checkpoints(epoch, train_loss, val_loss)
            
        # Wandb finish
        self.run.finish()
        
    def init_datasets(self, verbose: bool = True) -> tuple[DataLoader, ...]:
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
                                    save_format=self.cfg.dataset.save_format,
                                    filter_conditions=self.cfg.dataset.filter_conditions)
        
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

        if verbose:
            print(f"Training:\n\t- len: {len(self.dataset_train)}"+\
                  f"\n\t- # of minibatches: {len(self.train_loader)}")
            print(f"Validation:\n\t- len: {len(self.dataset_val)}"+\
                  f"\n\t- # of minibatches: {len(self.val_loader)}")
            print(f"Test:\n\t- len: {len(self.dataset_test)}"+\
                  f"\n\t- # of minibatches: {len(self.test_loader)}")

        return self.train_loader, self.val_loader, self.test_loader
    
    def save_config(self) -> None:
        if self.cfg.wandb.mode == "online":
            path = f'checkpoints/{self.begin_date}/run_{self.run.id}'
            os.makedirs(path, exist_ok=True)
            shutil.copyfile("config/trainer.yaml", path + "/trainer.yaml")
            
    def save_checkpoints(self,
                         epoch: int,
                         train_loss: torch.Tensor,
                         val_loss: Optional[torch.Tensor] = None) -> None:
        
        # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
        general_checkpoint = {"model_state_dict": self.model.state_dict(),
                              "optimizer_state_dict": self.optimizer.state_dict(),
                              "epoch": epoch,
                              "train_loss": train_loss}
        if val_loss is not None:
            general_checkpoint["val_loss"] = val_loss
        
        torch.save(general_checkpoint, "checkpoints/last_general_checkpoint.pth")
        
        # Also save it within particular folder related to wandb run.
        if self.cfg.wandb.mode == "online":
            path = f'checkpoints/{self.begin_date}/run_{self.run.id}'
            os.makedirs(path, exist_ok=True)
            
            torch.save(general_checkpoint, path + "/general_checkpoint.pth")

    def lowercase(self, txt: Optional[str]) -> Optional[str]:
        return txt.lower() if txt else None
    
    def create_model(self) -> nn.Module:
        """
        Create model based on self.cfg.model.type, e.g Multi Layer Perceptron 
        """
        match self.lowercase(self.cfg.model.type):
            case "mlp":
                inner_activation_fct = self.cfg.model.inner_activation_fct
                output_activation_fct = self.cfg.model.output_activation_fct
                hidden_layer_sizes = self.cfg.model.hidden_layer_sizes
                in_size = self.dataset_full.n_features
                
                layers = []
                for h in hidden_layer_sizes:
                    out_size = h
                    layers.append(nn.Linear(in_size, out_size))
                    match self.lowercase(inner_activation_fct):
                        case "relu":
                            layers.append(nn.ReLU())
                        case _:
                            raise NotImplementedError("Inner activation function"+\
                                                      f" {inner_activation_fct} not recognized")
                    in_size = out_size
                
                ## Last layer
                layers.append(nn.Linear(in_size, self.dataset_full.n_targets))
                # No activation function or identity by default (nn.Identity())
                match self.lowercase(output_activation_fct):
                    case _:
                        # layers.append(nn.Identity())
                        print("Using default identity activation"+\
                              " function for last layer")

                model = nn.Sequential(*layers)
            case _:
                raise NotImplementedError(f"Model type {self.cfg.model.type} not recognized")
        return model