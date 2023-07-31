## What are these files ?

- The name of the folder refers to the Weights & Biases run id and it can be used to retrieve the run on W&B.
  Ex with `2023-07-29/run_xfii9c1k`, you can go to: https://wandb.ai/stephane-nguyen/POLAR-background-prediction/runs/xfii9c1k

- The `trainer.yaml` files were copied from `../config` folder (can differ from current `../config/trainer.yaml`)
- The `general_checkpoint.pth` contains what you need to load the trained model
- The `last_general_checkpoint.pth` is what is used in when you call `../src/visualizer.py` (and potentially other files such as those in `../notebooks`)
