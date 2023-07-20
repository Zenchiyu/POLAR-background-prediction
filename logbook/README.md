# Logbook


## Week 1: 03.07.23 - 09.07.23

I transformed the dataset in the `.root` format into other formats which can be manipulated inside Python. I also preprocessed
the dataset in a certain way (not necessarily final). I visualized the dataset and what we wanted to predict.

We used a particular dataset with 6 million entries and is over 1 gb in size. We are in a regression problem and want to predict
photon rates from other measurements (magnetic field, latitude, longitude, cosmic rates etc.).

### Summary
- Getting started [ROOT CERN](https://root.cern/) because the data `Allaux_Bfield.root` is in `.root` format:
	- A few keywords related to ROOT: profile histogram, colz, TTree.
- Tried reproducing in ROOT, different plots Nicolas Produit showed me (profile histograms,
sum of cosmic rates over the whole mission for different latitude, longitude etc.).
- Tried using FFT in ROOT but had errors (even when using their [example](https://root.cern/doc/master/FFT_8C.html)).
- Tried to install missing dependencies (e.g FFTW3) but it didn't solve the errors
- Therefore, started to look at python libraries: [uproot](https://uproot.readthedocs.io/en/latest/basic.html), [ROOT or PyROOT](https://root.cern/manual/python/), 
and [root_numpy](http://scikit-hep.org/root_numpy/start.html).
- As they were other errors with PyROOT and root\_numpy, I chose to continue with uproot.

- Created functions to import `.root` files into pandas dataframes.
- Preprocessed the data `Allaux_Bfield.root` (dataset not in the github but see [this notebook for more information](https://github.com/Zenchiyu/POLAR-background-prediction/blob/develop/notebooks/exploring_polar_data.ipynb) and [this notebook](https://github.com/Zenchiyu/POLAR-background-prediction/blob/develop/notebooks/dataset.ipynb)) where I, for instance:
	- "quantized" the data so that the examples are at round seconds (each two seconds). Note that there are missing data so examples are not necessarily at equidistant times.
	- ignored/removed part of the data so that we work with a subset (e.g keep data after the period in which astronauts went onboard the space lab)
- Applied FFT on the time series: `sum_fe_rate`against "quantized" time (this is called the light curve). Note that because there can be some missing data,
it's not completely correct to use FFT. However, due to the [orbital period of Tiangong-2 space lab](https://en.wikipedia.org/wiki/Tiangong-2), Earth's rotation, etc., there are seasonalities involved and we could still observe a spike around "per 1 hour 30" frequency in the magnitude spectrum of the light curve.

- Using the results of FFT, we want to perform some operations in Fourier domain before reconstructing the light curve and using it as a target in our regression problem:
	- Started to code something in order to kill the spikes in the magnitude spectrum using two methods:
		- Using a box filter on the magnitude (but what window size should we use ? what padding method ?)
		- Applied linear regression in the "log x, log y" magnitude plot as the magnitude spectrum looked like some power law. We then wanted to use it to find the spikes before killing only the spikes (unfinished as we moved on to another idea, see next bullet point)
	- We stopped trying to kill the spikes and started to think about killing low frequencies instead as we're mostly interested in high frequencies
due to GRBs (Gamma Ray Bursts) which could cause visible spikes:
		- We manually chose some threshold based on the magnitude spectrum to kill some low frequencies as well as some spikes

- We won't necessarily use the reconstructed light curve as target
- Setted up the github project and pipenv

### (Future) Goals:
- To better understand how to split the data into train, validation test set.
- To try some simple model to predict `sum_fe_rate` from all the other measurements (magnetic field, latitude, longitude, etc.). It's as if
we're predicting a time series or sequence using multiple time series or sequences.


## Week 2: 10.07.23 - 16.07.23

- `fe_rate` contains $25$ values representing photon rates from different modules but `rate` contains $12$ values representing photon rates for different "energies" (but I don't know what they are as I'm not the expert).
- We used linear regression for two datasets, the same as previous week as well as a new one `fm_rate`. Therefore we also loaded `fm_rate`, preprocessed it etc.
- The datasets we use are not necessarily the final ones.
- For the data `fm_rate` inputted to the neural network, we tried with different features (e.g all measurements except targets and `unix_time`).

### Summary:
- Visualized the Pearson correlation coefficient between the measurements (magnetic field, latitude, longitude, cosmic rates etc.) as well as
with our target. Found that cosmic rates have quiet some linearly correlation with our photon rates (target) even though it's not sufficient !

- Applied linear regression (see this [notebook](https://github.com/Zenchiyu/POLAR-background-prediction/blob/develop/notebooks/linear_regression.ipynb)) using only cosmic rates in order to predict photon rates and found that, depending on how we split the dataset:
	- Take the whole dataset as training set: We observe very good predictions (visually) except for some huge spikes (no validation, test set so it was already a bad thing to do)
	- Randomly shuffle the dataset then split 60 \% train, 20 \% validation, 20 \% test. We ignore completely the temporal dependencies and work with out data as if examples are i.i.d ..: We observe quiet "bad" validation set predictions (visually).

- We used another dataset `f1_rate` and applied similar steps as the dataset from previous week. However, as there were missing things in this dataset, we stopped using it.
- We then started using another dataset `fm_rate` and applied similar steps as the dataset from previous week. Note however that this dataset comprises of only about 60k examples, with intervals of about 60 seconds
between them (except for missing data or 'holes'). The `m` comes from "m"inute.

- Using that dataset (splitting it 60/20/20 for train, validation, test after shuffling), we tried applying a simple fully connected neural network from sklearn using the base MLPRegressor but with 100 neurons in the hidden layer.
We moved on to two hidden layers with 100 neurons each (see this [notebook](https://github.com/Zenchiyu/POLAR-background-prediction/blob/develop/notebooks/fmrate_prediction.ipynb)). Instead of predicting the sum of rates obtained from each module, we try to predict each rates from "each energy" (`rate[i]` instead of `sum_fe_rate`)
- With similar data split, we tried applying a linear regression to predict `rate[0]` only using `sum_fe_cosmic`
- Even though **it's incorrect** to use the whole dataset, we used our trained model to predict over the whole dataset, the photon rates `rate[0]`.
- From them, we computed the residual plots (target-prediction), showed their histograms, gaussian fits of residuals.
- We also showed rescaled residual plots (target-prediction)/sqrt(target) ("pull" plot), their histograms and modified gaussian fits of "pulls". The modified gaussian fit:

```
def find_std(data):
    low = -np.inf
    high = np.inf
    prev_std = np.inf
    std = np.std(data)
    mean = np.mean(data)
    
    while ~np.isclose(prev_std, std):
        # Update interval
        low = -3*std + mean
        high = 3*std + mean
        
        prev_std = std
        std = np.std(data[(data>low) & (data<high)])
        print(mean, std, low, high)
    return mean, std
```

was suggested by Nicolas Produit in order to ignore the "outliers" in the "pull histogram".

- Started learning about pytorch, weights and biases and JAX.
- Issues installing JAX with GPU support on Windows (my desktop computer), therefore, stayed with Pytorch with GPU.
- Modified README.md by adding information about how to use pipenv and how to install.

### (Future) Goals:
- To better understand how to split the data into train, validation test set for our application as they are maybe some 'issues' related to overfitting when we shuffle our data
and pick train, validation, test set where examples can be close to each others in time (or other measurements). We maybe want to also take into account
temporal relationships.
- To try using more complex models to predict photon rates from all the other measurements (magnetic field, latitude, longitude, etc.). It's as if
we're predicting a time series or sequence using multiple time series or sequences (something to explore).
- To try using pytorch and GPUs

## Week 3: 18.07.23 - 23.07.23


### Summary

- Started writing logbook
- Connected to GPU (Quadro RTX 4000) of POLAR group. Can run my python scripts remotely.
- Started learning about weights and biases and using it for the first time ([Project's weights and biases](https://wandb.ai/stephane-nguyen/POLAR-background-prediction?workspace=user-stephane-nguyen)). 
Here's an example of [run](https://wandb.ai/stephane-nguyen/POLAR-background-prediction/runs/1j329ps1?workspace=user-stephane-nguyen).
- Started writing the pytorch code with GPU support (device) taking inspiration from https://github.com/eloialonso/iris project (started using hydra
for first time too).
- Added code to save models, criterions and more
- Applied model on validation set and visualized prediction (over whole validation set)

### Comments

- pipenv currently "half-outdated" due to hydra and pytorch gpu (pipenv not working as expected on the POLAR machine, couldn't install pytorch via pipenv but
installed it via normal pip)

- Run:
```
python src/main.py
```
to run the training phase and log information in Weights and Biases.

- Run:
```
python src/main.py wandb.mode=disabled
```
to run the training phase without logging information into Weights and Biases.


### (Future) Goals:
- To better understand how to split the data into train, validation test set for our application as they are maybe some 'issues' related to overfitting when we shuffle our data
and pick train, validation, test set where examples can be close to each others in time (or other measurements). We maybe want to also take into account
temporal relationships.
- To try using more complex models to predict photon rates from all the other measurements (magnetic field, latitude, longitude, etc.). It's as if
we're predicting a time series or sequence using multiple time series or sequences (something to explore).
- To better understand Adam optimizer, different parts of what I've used in general.
- To better understand or to learn more about Hydra
- To use W&B artifacts for datasets. Need to version datasets as I can work with different datasets
- To learn more about regularization, dropout, batch normalization
- To learn more about W&B sweeps and add more logs information.
