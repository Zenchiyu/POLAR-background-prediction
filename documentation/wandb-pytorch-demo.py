# Demo made by Nguyen Stephane Liem
import time
import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm


def computations(run, device):
    # PyTorch tensor on device
    x = torch.linspace(config["Tmin"], config["Tmax"], config["N"],
                       device=device)
    y = config["f"](x)  # operations are on device
    print(f"x.device: {x.device}, y.device: {y.device}")
    plt.plot(x, y)
    plt.show()
    plt.savefig("demo.png")

    # Dummy PyTorch Model (module)
    model = nn.Sequential(nn.Linear(1, 1))

    # XXX: wandb track model too
    run.watch(model, log="all", log_graph=True)

    # Dummy Loop where I can log information on wandb
    for i in tqdm(torch.arange(100)):
        wandb.log({
            "i": i,
            "cos(i/100*2*pi)": torch.cos(i/100*2*torch.pi),
            "img": wandb.Image("demo.png")
        })
        time.sleep(5)  # just to make it extra slow

    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = {
        "device": device,
        "Tmin": 20,
        "Tmax": 100,
        "N": 10,
        "f": lambda x: x**2
    }

    # XXX: WANDB INIT
    run = wandb.init(
        project="wandb-pytorch-demo",
        config=config
    )

    computations(run, device)

    # XXX: WANDB FINISH
    wandb.finish()


