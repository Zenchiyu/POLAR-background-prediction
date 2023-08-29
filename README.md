[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# POLAR background prediction

The idea of the project is to first predict the background part of a light curve (y axis being the number of photons arriving into the POLAR detector per second) or transformed light curve. After fitting the curve, we would like to subtract it from the original curve and use the resulting curve to detect gamma ray bursts (GRBs).

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

The code was developed for Python `3.10.12` and `3.10.6` and with torch==2.0.1 and torchvision==0.15.2.

## Usage

- `python src/main.py` to run the training phase (use `python src/main.py wandb.mode=disabled` if don't want to use weights and biases)
- `python src/visualizer.py` to load pretrained model, plot loss and predicted photon rates for validation set.
- After running the previous command, you can run the jupyter notebook `./notebooks/results.ipynb` to show our other plots.

You can change the `config/trainer.yaml` if you want a different model architecture, different hyperparameters, features etc.

## Credits

- Code and project structure: https://github.com/eloialonso/iris
- 55 GRBs: https://www.researchgate.net/publication/326811280_Overview_of_the_GRB_observation_by_POLAR
