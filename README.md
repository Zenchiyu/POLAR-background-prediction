[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# POLAR background prediction

We first try to predict the background part of a light curve (y axis being the number of photons arriving into the POLAR detector per second) or transformed light curve. We would then like to subtract the prediction from the original and use the resulting curve to detect gamma ray bursts (GRBs).

| <img src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/f2fa9896-db10-4742-b824-1cbe684a8b59" width=300> |
|:--:| 
| *Trying to detect 25 known GRBs: red dot: above threshold, green vertical line: trigger time GRB, blue curve: original curve (e.g. `rate[0]`), black curve: predicted curve* |


Related links:
- https://www.astro.unige.ch/polar/
- https://www.unige.ch/dpnc/fr/grups/xin-wu/experiences/polar/

---

## Installation

- Install pytorch:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
- And other dependencies:
```
pip3 install -r requirements.txt
```

The code was developed for Python `3.10.12` and `3.10.6` and with torch==2.0.1 and torchvision==0.15.2. Using a different Python version might cause problems.

I also used Jupyter `v2023.7.1002162226` and Remote-SSH `v0.102.0` extensions of VSCode `1.81.1` to remotely edit and run codes on raidpolar (POLAR research group's Linux machine).

## Usage

- `python src/main.py` to run the training phase (use `python src/main.py wandb.mode=disabled` if don't want to use weights and biases)
- `python src/visualizer.py` to load pretrained model, plot loss and predicted photon rates for validation set.
- Run the different cells of `./notebooks/results.ipynb` to show the rest of our (interactive) plots (clusters, cluster intersections, etc.).

You can change the `config/trainer.yaml` if you want a different model architecture, hyperparameters, features etc. To remotely run our Python scripts without keeping an opened SSH connection for the whole execution duration, you can use `tmux` and detach the session.

## Credits and useful links

- Code and project structure: https://github.com/eloialonso/iris
- 55 GRBs: https://www.researchgate.net/publication/326811280_Overview_of_the_GRB_observation_by_POLAR
- Weights & biases: https://wandb.ai/site
- Pytorch:
  - https://pytorch.org/tutorials/beginner/basics/intro.html
  - https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- Hydra: https://hydra.cc/docs/intro/
- ReviewNB ("git diff but for Jupyter notebooks"): https://www.reviewnb.com/
- VSCode Remote SSH: https://code.visualstudio.com/docs/remote/ssh-tutorial
- Diff with colors by using `diff <file1> <file2> --color`: https://man7.org/linux/man-pages/man1/diff.1.html
- TMux detach/reattach session: https://www.redhat.com/sysadmin/introduction-tmux-linux
