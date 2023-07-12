# POLAR background prediction

The idea of the project is to first predict the background part of a light curve (y axis being the number of photons arriving into the POLAR detector per second) or transformed light curve. After fitting the curve, we would like to subtract it from the original curve and use the resulting curve to detect gamma ray bursts (GRBs).

# Installation
If you don't use Anaconda, you can skip the first steps with Anaconda but you need to make sure you already have Python version `3.10.12`. Moreover, Jupyter notebook is required to run the notebooks. 

## Preparing an Anaconda environment for pipenv

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

## Pipenv

All you need should be to do `pipenv shell`

## How to use jupyter notebook with pipenv ?

```
python -m ipykernel install --user --name=polar-virtualenv
```

- https://stackoverflow.com/questions/47295871/is-there-a-way-to-use-pipenv-with-jupyter-notebook
