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

### FAQ

<details>
<summary>I want to use a different model architecture, hyperparameters, features etc. What can I do?
</summary>
<br>

You can change the `config/trainer.yaml`. However, your possiblities are limited to what I've implemented. Please refer to the documentation for more information.
</details>

<details>
<summary>Can I run src/main.py and src/visualizer.py remotely? </summary>
<br>

Yes, you can. To remotely run our Python scripts without keeping an opened SSH connection for the whole execution duration, you can use `tmux` and detach the session.
</details>

<details>
<summary>Can I run the Jupyter notebooks remotely?</summary>
<br>

Yes, you can. You can use Jupyter and Remote-SSH VSCode extensions. They allow you to edit and run codes on your remote Linux machine.

If you don't want to use VSCode, you can take a look at the following link:
https://docs.anaconda.com/free/anaconda/jupyter-notebooks/remote-jupyter-notebook/
</details>

<details>
<summary>The execution crashed. What happened?</summary>
<br>

The crash is likely due to memory usage as we sometimes create/store data structures with over 3 million entries.

- You can change `verbose: False` to `verbose: True` in `config/trainer.yaml` to see more information (our prints).
- You can check `cfg.common.device` in `config/trainer.yaml`, you might need to change it to `cpu` if you don't have a GPU (you can check that using `torch.cuda.is_available()` in Python).
- You can use `nvidia-smi` to see the VRAM usage (if you're using a GPU)
- You can use `htop` (or another command) to see the RAM usage.
- I sometimes create tensors containing the whole dataset to perform a single forward pass on all the examples instead of many.
In terms of memory usage, this is not great. Instead, you can try to work with mini-batches despite the fact that you will have to perform multiple forward passes.

</details>

<details>
<summary>Can I run the codes on CPU?</summary>
<br>

- `src/main.py`: It is recommended to use the GPU for model training as it's faster. Moreover, as I only tried once to train my model using the research group's CPU, it's uncertain if it works as expected. However, if you still want to change the device to CPU, you need to change `cfg.common.device: cuda` to either `cfg.common.device: cpu`.
- `src/visualizer.py` and `notebooks/results.ipynb` work by default on GPU (if available, otherwise work on CPU) and overrides whatever you've selected via `cfg.common.device`. However, you can still manually choose the device by changing in the code:

```python
cfg.common.device = "cuda" if torch.cuda.is_available() else "cpu"
```
to
```python
cfg.common.device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
```

before `trainer = Trainer(cfg)`.

</details>

<details>
<summary>Why results using CPU are different than when using GPU?</summary>
<br>

See 
https://discuss.pytorch.org/t/why-different-results-when-multiplying-in-cpu-than-in-gpu/1356/6
</details>

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
