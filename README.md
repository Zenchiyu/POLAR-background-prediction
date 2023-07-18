[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# POLAR background prediction

The idea of the project is to first predict the background part of a light curve (y axis being the number of photons arriving into the POLAR detector per second) or transformed light curve. After fitting the curve, we would like to subtract it from the original curve and use the resulting curve to detect gamma ray bursts (GRBs).

Related links:
- https://www.astro.unige.ch/polar/
- https://www.unige.ch/dpnc/fr/groups/xin-wu/experiences/polar/

---

## Installation (OUTDATED - need to take into account pytorch and hydra and the case where we only use pip)
If you don't use Anaconda, you can skip the first steps with Anaconda but you need to make sure you already have Python version `3.10.12` and `pipenv` installed (using `pip` for ex.). Moreover, Jupyter notebook is required to run the notebooks. 


<details>
<summary>Preparing an Anaconda environment for pipenv</summary>
  
The anaconda environment is mostly used here to have the Python version `3.10.12`.

1. You can create a new Anaconda environment and activate the environment.

```
conda create --name polar && conda activate polar
```

2. Install Python version `3.10.12`

```
conda install python=3.10.12
```

3. And then install pipenv (and jupyter notebook)

```
conda install pipenv && conda install jupyter
```

</details>

- **All in one**:
```
conda create -y --name polar&& conda activate polar && conda install -y python=3.10.12 && conda install -y pipenv && conda install -y jupyter 
```

- The [following](https://stackoverflow.com/questions/36345136/use-unix-based-commands-with-anaconda-in-windows-operating-system) might also be useful if you want to use commands such as `ls` within your conda environment:
```
conda install m2-base
```


<details>
<summary>Pipenv</summary>

All you need should be to do `pipenv install` then `pipenv shell`

</details>

<details>
<summary>How to use jupyter notebook with pipenv ?</summary>
  
```
python -m ipykernel install --user --name=polar-virtualenv
```

```
jupyter notebook
```
Then inside jupyter notebook, select `polar-virtualenv` kernel.



- See this link for more details: https://stackoverflow.com/questions/47295871/is-there-a-way-to-use-pipenv-with-jupyter-notebook

</details>

## Usage

Assuming you already did the installation steps. There are different main scenarios:

<details>
<summary>Want to use jupyter notebooks with Anaconda</summary>

```
conda activate polar && pipenv shell
```
```
jupyter notebook
```

Then inside jupyter notebook, select `polar-virtualenv` kernel.

</details>

<details>
<summary>Want to use jupyter notebooks without Anaconda</summary>

```
pipenv shell
```

```
jupyter notebook
```

Then inside jupyter notebook, select `polar-virtualenv` kernel.

</details>

<details>
<summary>Run python scripts</summary>

```
pipenv shell
```

Then `python <script-name>`.

</details>

- `python src/main.py` to run the training phase (use `python src/main.py wandb.mode=disabled` if don't want to use weights and biases)
- `python src/visualizer.py` to load pretrained model, plot loss and predicted photon rates for validation set.

## Credits

A part of the code was inspired from https://github.com/eloialonso/iris
