# Logbook


## Week 1: 03.07.23 - 09.07.23

<details>

I transformed the dataset in the `.root` format into other formats which can be manipulated inside Python. I also preprocessed the dataset in a certain way (not necessarily final). I visualized the dataset and what we wanted to predict.

We used a particular dataset with 6 million entries and is over 1 gb in size. We are in a regression problem and want to predict photon rates from other measurements (magnetic field, latitude, longitude, cosmic rates etc.).

### Summary
- Getting started with [ROOT CERN](https://root.cern/) because the data `Allaux_Bfield.root` is in `.root` format:
	- A few keywords related to ROOT: profile histogram, colz, TTree.
- Tried reproducing in ROOT, different plots Nicolas Produit showed me (profile histograms,
sum of cosmic rates over the whole mission for different latitudes, longitudes etc.).
- Tried using FFT in ROOT but had errors (even when using their [example](https://root.cern/doc/master/FFT_8C.html)).
- Tried to install missing dependencies (e.g FFTW3) but it didn't solve the errors
- Therefore, started to look at Python libraries: [uproot](https://uproot.readthedocs.io/en/latest/basic.html), [ROOT or PyROOT](https://root.cern/manual/python/), and [root_numpy](http://scikit-hep.org/root_numpy/start.html).
- As they were complications with PyROOT and root\_numpy, I chose to continue with uproot.

- Created functions to import `.root` files into pandas dataframes.
- Preprocessed the data `Allaux_Bfield.root` (dataset not in the GitHub but see [this notebook for more information](https://github.com/Zenchiyu/POLAR-background-prediction/blob/develop/notebooks/exploring_polar_data.ipynb) and [this notebook](https://github.com/Zenchiyu/POLAR-background-prediction/blob/develop/notebooks/dataset.ipynb)) where I, for instance:
	- "quantized" the data so that the examples are at round seconds (each two seconds). Note that there are missing data so examples are not necessarily at equidistant times.
	- ignored/removed part of the data so that we work with a subset (e.g. keep data after the period in which astronauts went onboard the space lab)
- Applied FFT on the time series: `sum_fe_rate` against "quantized" time (this is called the light curve). Note that because there can be some missing data, it's not completely correct to use FFT. However, due to the [orbital period of the Tiangong-2 space lab](https://en.wikipedia.org/wiki/Tiangong-2), Earth's rotation, etc., there are seasonalities involved and we could still observe a spike around "per 1 hour 30" frequency in the magnitude spectrum of the light curve.

- Using the results of FFT, we want to perform some operations in the Fourier domain before reconstructing the light curve and using it as a target in our regression problem:
	- Started to code something in order to kill the spikes in the magnitude spectrum using two methods:
		- Using a box filter on the magnitude (but what window size should we use ? What padding method ?)
		- Applied linear regression in the "log x, log y" magnitude plot as the magnitude spectrum looked like some power law. We then wanted to use it to find the spikes before killing only the spikes (unfinished as we moved on to another idea, see next bullet point)
	- We stopped trying to kill the spikes and started to think about killing low frequencies instead as we're mostly interested in high frequencies
due to GRBs (Gamma Ray Bursts) which could cause visible spikes:
		- We manually chose a threshold based on the magnitude spectrum to kill some low frequencies as well as some spikes

- We won't necessarily use the reconstructed light curve as target
- Set up the GitHub project and pipenv

### (Future) Goals:
- To better understand how to split the data into train, validation test set.
- To try some simple model to predict `sum_fe_rate` from all the other measurements (magnetic field, latitude, longitude, etc.). It's as if
we're predicting a time series or sequence using multiple time series or sequences.

</details>

## Week 2: 10.07.23 - 16.07.23

<details>

- `fe_rate` contains $25$ values representing photon rates from different modules but `rate` contains $12$ values representing photon rates for different "energies" (but I don't know what they are as I'm not the expert).
- We used linear regression for two datasets, the one from last week as well as a new one `fm_rate`. Therefore we also loaded `fm_rate`, preprocessed it etc.
- The datasets we use are not necessarily the final ones.
- For the data `fm_rate` inputted to the neural network, we tried with different features (e.g. all measurements except targets and `unix_time`).

### Summary:
- Visualized the Pearson correlation coefficient between the measurements (magnetic field, latitude, longitude, cosmic rates etc.) as well as with our target. Found that cosmic rates have quite some linear correlation with our photon rates (target) even though it's not sufficient!

- Applied linear regression (see this [notebook](https://github.com/Zenchiyu/POLAR-background-prediction/blob/develop/notebooks/linear_regression.ipynb)) using only cosmic rates in order to predict photon rates and found that, depending on how we split the dataset:
	- Take the whole dataset as the training set: We observe very good predictions (visually) except for some huge spikes (no validation, test set so it was already a bad thing to do)
	- Randomly shuffle the dataset then split 60 \% train, 20 \% validation, and 20 \% test. We ignore completely the temporal dependencies and work with data as if examples are i.i.d ..: We observe quiet "bad" validation set predictions (visually).

- We used another dataset `f1_rate` and applied similar steps as the dataset from last week. However, as there were missing things in this dataset, we stopped using it.
- We then started using another dataset `fm_rate` and applied similar steps as the dataset from previous week. Note however that this dataset comprises only about 60k examples, with intervals of about 60 seconds
between them (except for missing data or 'holes'). The `m` comes from "m"inute.

- Using that dataset (splitting it 60/20/20 for train, validation, and test after shuffling), we tried applying a simple fully connected neural network from sklearn using the base MLPRegressor but with 100 neurons in the hidden layer.
We moved on to two hidden layers with 100 neurons each (see this [notebook](https://github.com/Zenchiyu/POLAR-background-prediction/blob/develop/notebooks/fmrate_prediction.ipynb)). Instead of predicting the sum of rates obtained from each module, we try to predict each rate from "each energy" (`rate[i]` instead of `sum_fe_rate`)
- With a similar data split, we tried applying linear regression to predict `rate[0]` only using `sum_fe_cosmic`
- Even though **it's incorrect** to use the whole dataset, we used our trained model to predict over the whole dataset, the photon rates `rate[0]`.
- From them, we computed the residual plots (target-prediction), showed their histograms, and Gaussian fits of residuals.
- We also showed rescaled residual plots (target-prediction)/sqrt(target) ("pull" plot (particle physics jargon)), their histograms and modified Gaussian fits of "pulls". The modified Gaussian fit:

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

was suggested by Nicolas Produit to ignore the "outliers" in the "pull histogram".

- Started learning about PyTorch, weights and biases and JAX.
- Issues installing JAX with GPU support on Windows (my desktop computer), therefore, stayed with Pytorch with GPU.
- Modified README.md by adding information about how to use pipenv and how to install it.

### (Future) Goals:
- To better understand how to split the data into train, validation test set for our application as they are maybe some 'issues' related to overfitting when we shuffle our data and pick train, validation, and test set where examples can be close to each other in time (or other measurements). We maybe want to also take into account temporal relationships.
- To try using more complex models to predict photon rates from all the other measurements (magnetic field, latitude, longitude, etc.). It's as if we're predicting a time series or sequence using multiple time series or sequences (something to explore).
- To try using PyTorch and GPUs

</details>


## Week 3: 17.07.23 - 23.07.23

<details>

### Summary

- Started writing logbook
- Connected to GPU (Quadro RTX 4000) of POLAR group. Can run my Python scripts remotely (and used tmux to run my codes without the need for my computer to be on).
- Started learning about "weights and biases" tool and using it for the first time ([Project's weights and biases](https://wandb.ai/stephane-nguyen/POLAR-background-prediction?workspace=user-stephane-nguyen)). 
Here's an example of a [run](https://wandb.ai/stephane-nguyen/POLAR-background-prediction/runs/1j329ps1?workspace=user-stephane-nguyen).
- Started writing the PyTorch code with GPU support (device) taking inspiration from https://github.com/eloialonso/iris project (started using Hydra for the first time too).
- Added code to save models, criteria and more
- Applied model on validation set and visualized prediction (over whole validation set)
- Further cleaning of code and added Python type hints (not for all files though)
- Can now save a general checkpoint at two different places; one as the last checkpoint and the other is attached to a date and run id (see checkpoints folder)
- Can now specify the number of neurons for each hidden layer directly inside the yaml config file.
- Removed pipenv, we no longer use pipenv. Modified README in consequence.
- Trained model again but on `nf1rate` (taking about 3 hours for training) with as target `rate[0]` (using all training examples, no additional filtering based on `rate_err[0]`) ([see wandb run](https://wandb.ai/stephane-nguyen/POLAR-background-prediction/runs/3zdzy861?workspace=user-stephane-nguyen)).
- Trained model again on "same" dataset but with as target `rate[0]/rate_err[0]` (filtered examples when cannot divide) ([see wandb run](https://wandb.ai/stephane-nguyen/POLAR-background-prediction/runs/3hevg2jy/overview?workspace=user-stephane-nguyen))
- Added more plots in `src/visualizer` where we can now plot the residual plot with its histogram.

### Comments

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
- To better understand how to split the data into train, validation test set for our application as they are maybe some 'issues' related to overfitting when we shuffle our data and pick train, validation, and test set where examples can be close to each other in time (or other measurements). We maybe want to also take into account temporal relationships.
- To read more about predicting a time series or sequence using multiple time series or sequences (something to explore).
- To better understand Adam optimizer, different parts of what I've used in general.
- To better understand or to learn more about Hydra
- To use W&B artifacts for datasets. Need to version datasets as I can work with different datasets
- To learn more about regularization, dropout, batch normalization
- To learn more about W&B sweeps and add more log information.

</details>

## Week 4: 24.07.23 - 30.07.23

<details>

### Summary

- Exploring the 55 GRBs (from [Overview_of_the_GRB_observation_by_POLAR's paper](https://www.researchgate.net/profile/Yuanhao-Wang-8/publication/326811280_Overview_of_the_GRB_observation_by_POLAR/links/5cfe12c0a6fdccd1308f8b32/Overview-of-the-GRB-observation-by-POLAR.pdf), after converting UTC to Unix time) and comparing them to our dataset:

<p align="center">
<img src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/cd6024cc-37ed-4a7b-a8a2-774cd53c8a99" width=300>
</p>

We can observe that there are GRBs (in red) outside the time range (both to the left and the right) of our dataset (in blue)

- Only restricting to our time range, we're left with 25 GRBs:

<p align="center">
<img src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/4013d962-4b2f-48ec-8bdc-09595a1a195d" width=300>
</p>

A closer look (+- 50 seconds windows):

<p align="center">
<img src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/cf89a4da-2484-40db-bcdb-3b1e6400bf33" width=300>
</p>

Note that the one at the bottom-mid was within the period with no data.

- From the residual histogram (from applying our model to the validation set) and modified Gaussian fit, we highlighted the data points from the validation set having
their residual above 5 standard deviation:

<p align="center">
<img src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/36a33a27-afde-4c81-9c8c-18b2d6b59ac9" width=300>
<img src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/666e62c7-1f41-455a-a65a-bba77cbf6365" width=300>
</p>

We also showed in blue the full dataset (train + validation + test) even though we "shouldn't". There are 9980 red points.

- If we compare the red points with the 25 GRBs, we can only see $5$ red points. Moreover, we must remember the fact that we're showing red points that are from the validation set, not the full dataset.

<p align="center">
<img src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/5890e455-8e5c-46c8-967c-8200274d9635" width=300>
</p>

- Fixed create_columns where it could try to create, for instance, a column based on a `data_df["<numerical value>"]` which was not intended.
- Added `filter_conditions` to the YAML and modified Python code to filter examples based on `filter_conditions`
- Ran the training phase with a filtered dataset where we only keep examples having `rate[0]/rate_err[0]` greater than 20. It gives this:

<p align="center">
<img src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/094c849f-c217-4c35-a15a-df7e7768f6a6" width=300>
</p>

where again the red points come from the validation set and have residuals > 5 standard deviations (recall that when we say standard deviation, we talk about the modified one based on the modified Gaussian fit).

- Ran the training phase again but ignored +-100 seconds around the 25 GRBs. Also ignored them in the validation and test set but maybe shouldn't because we
no longer can compare the prediction for these +-100 seconds around the 25 GRBs with the real curve. We can't plot anymore the plot we've shown above. However, here's a zoomed-in version of what our model predicts in 4 arbitrary intervals of the validation set:

<p align="center">
<img src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/af0b8810-d791-48cd-b480-175d0430049d" width=300>
</p>

`l` and `h` are indices. For instance, if `l=0`, then it means we show `h` first validation set examples (ordered by ascending time). In red we have the prediction, and in green, the validation set.

- By cleaning the code, I discovered that I was training on the validation set unintentionally, I fixed it and then ran the training phase again. I show below
the previous plot but with the fixed code:

<p align="center">
<img src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/c01d11ab-cb09-494b-848e-ee38de9a73cf" width=300>
</p>

- Plotting prediction over train + validation set in red. In blue/cyan we have the training set and in green, we have the validation set

<p align="center">
<img src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/c03301bb-a31f-40ea-8b93-ad6c9082882e" width=300>
</p>

We can observe that it doesn't overfit severely but it might still overfit...

- Started reading a bit about unsupervised learning anomaly detection with autoencoders and using the reconstruction error to detect anomalies:
	- https://keras.io/examples/timeseries/timeseries_anomaly_detection/
   	- https://towardsdatascience.com/using-lstm-autoencoders-on-multidimensional-time-series-data-f5a7a51b29a1
	- https://youtu.be/6S2v7G-OupA
- Started reading a bit about anomaly detection in general. I should maybe focus on semi-supervised anomaly detection:
	- https://ai.googleblog.com/2023/02/unsupervised-and-semi-supervised.html
	- https://arxiv.org/pdf/1906.02694.pdf
	- https://en.wikipedia.org/wiki/Anomaly_detection
> Semi-supervised anomaly detection techniques assume that some portion of the data is labelled. This may be any combination of the normal or anomalous data, but more often than not the techniques construct a model representing normal behavior from a given normal training data set, and then test the likelihood of a test instance to be generated by the model.
- Started reading a bit about time series regression. We need to analyze the auto-correlation function of residuals to see if there are correlated errors.
- Ran training for a different target; `rate[0]`. Also, instead of plotting the residuals, we plot the residuals divided by `rate_err[0]`. Filtering is the same as before and the plots have comparable/similar meanings to before (except for residuals and the target):


| | | |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/0c1d6735-97d0-4cc8-936c-b7cbe0e75e36"> Prediction over validation set in red|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/aaa0b2cb-029a-4209-9a62-605441d86c02"> A closer look at 4 intervals|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/370f9273-84df-4054-ab96-421b6d1d14ea"> Prediction over train + val, closer look|
<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/7e32e010-39e7-48d7-bfe7-c03212fcc5cf"> `(rate[0]-pred)/rate_err[0]`|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/78dbae85-7772-4ef1-ae54-c29258476c9c"> `(rate[0]-pred)/rate_err[0]` hist*|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/cf6ee699-6720-49de-8d3b-c29805222a37"> zoomed-in version|

\*: x-axis label should be "pull".
<!-- https://gist.githubusercontent.com/trusktr/93175b620d47827ffdedbf52433e3b37/raw/e980fa9116cb28dfbdee0dc5c17adc5ed91df783/image-grid.md -->

- If we use our trained model and apply it to the full dataset (train + val + test) including the 25 GRBs we removed, we can observe these:


| | |
|:-------------------------:|:-------------------------:|
|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/6ba9d6c8-9e5a-4a49-84d0-2637f0148ba6">|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/37175a7a-a89b-49f9-a8ff-7d6f86cd77ba">*|
<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/20ea06bf-b93d-4929-af4d-b903ed388d1f">|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/fdc21464-0c18-4945-b1af-b92f9ba5abec">|

\*: x-axis label should be "pull".

- Split differently the data in a periodical manner: train, validation, and test (120, 40, 40 data points) then train, validation, and test again (do it until no more data is left) (this time, the splitting is no longer random but there's still shuffling=True in the train loader and we still have 60 %, 20 %, 20 % split ratios):



| | | |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/0bf32195-3438-4531-8e9f-06c4e42e2869"> Prediction over validation set in red|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/aa2a6095-8394-42ee-bf2e-c34abf399326"> A closer look at 4 intervals|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/ca7ba5a9-c478-426f-9b86-9984db41f205"> Prediction over train + val, a closer look|
<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/a2f37eaa-8b74-4bb7-99ad-f6652328ffb1"> `(rate[0]-pred)/rate_err[0]`|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/9b82a87c-3dbc-4fc6-83ad-678234592213"> `(rate[0]-pred)/rate_err[0]` hist|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/da6267bd-cd97-42fb-b094-a53b89f09260"> Losses (average mini-batch MSE loss)|

- We can show how the losses behave compared to before (violet: `periodical_split`, yellow: `random_split`):

<p align="center">
<img src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/6d056ebd-b38a-4a0f-8db5-31d7236bc5a8" width=300>
</p>

And it shows more clearly the gap between train and validation losses.

- And if we use our trained model with this "periodical split" dataset and apply it to the full dataset (train + val + test) including the 25 GRBs we removed, we can observe these:


| | |
|:-------------------------:|:-------------------------:|
|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/c9d6670e-25c0-4f62-a842-b176e3f2795c">|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/4455e55e-de08-4554-b93b-4d6bdae5cb47">|
<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/d0a5370d-7d4d-4167-a08f-e8e9c0ece41c">|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/08cd8e81-4239-40e4-aaf6-d9a32856c521">|


### (Future) Goals:
- To better understand how to split the data into train, validation test set for our application as they are maybe some 'issues' related to overfitting when we shuffle our data and pick train, validation, and test set where examples can be close to each other in time (or other measurements). We maybe want to also take into account
temporal relationships. There's maybe something called "overfitting in feature space".

- Some links on splitting but our goal is not to forecast but to predict the "present" from the "present" (or maybe even past but not yet):
	- https://stats.stackexchange.com/questions/346907/splitting-time-series-data-into-train-test-validation-sets
	- https://datascience.stackexchange.com/questions/91162/why-is-shuffling-timeseries-a-bad-thing
- To read more about predicting a time series or sequence using multiple time series or sequences (something to explore) (and correlated residuals):
	- https://otexts.com/fpp2/regression.html
	- https://ethz.ch/content/dam/ethz/special-interest/math/statistics/sfs/Education/Advanced%20Studies%20in%20Applied%20Statistics/course-material-1921/Zeitreihen/ATSA_Script_v200504.pdf (from page 133)
- To better understand Adam optimizer, different parts of what I've used in general.
- To better understand or to learn more about Hydra
- To use W&B artifacts for datasets. Need to version datasets as I can work with different datasets
- To learn more about regularization, dropout, batch normalization
- To add a "stagnation end condition" to my training loop
- Is it fine to apply prediction over the whole dataset and threshold residuals to see whether known GRBs are part of them ? (and what if we apply unsupervised learning outlier detection over the residuals ?)

</details>

## Week 5: 31.07.23 - 06.08.23

<details>

### Summary

- Discovered that all this time, I sorted time in descending order... By fixing it, it fixed the issue with `GRB_170114A` that was not detected by thresholding the residuals. Note that this fix didn't affect the trained model, it only affects the visualized results.

#### Plots with target: `rate[0]` and with correct time sorting:

| | | |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/4d17bb6f-5bb9-4d0e-8bf5-657bb14e1e90"> Prediction over validation set in red|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/22e4c5e0-3861-4bb2-bbb1-faf2f9e2a2c3"> A closer look at 4 intervals|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/cb38a62f-35db-4f3d-82f5-576e14d0cae2"> Prediction over train + val, a closer look|
<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/edd18c95-419f-4c27-be63-5c9952b6da42"> `(rate[0]-pred)/rate_err[0]`|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/329e7144-a314-49ff-9078-60b445961026"> `(rate[0]-pred)/rate_err[0]` hist|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/da6267bd-cd97-42fb-b094-a53b89f09260"> Losses (average mini-batch MSE loss)|

- And if we use our trained model with this "periodical split" dataset and apply it to the full dataset (train + val + test) including the 25 GRBs we removed, we can observe these:


| | |
|:-------------------------:|:-------------------------:|
|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/74501fd8-2a5f-4953-b7c3-ab83a00572f5">|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/b99af059-a9f2-4e62-87bc-cfac49483514">|
<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/d407e4c5-381b-47d5-8bf2-373feddfa132"> 44553 red dots|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/33c330d8-eb03-46aa-9f18-68623ef26a0e">|

#### Plots with target: `rate[0]/rate_err[0]` and with correct time sorting:

| | | |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/343899fc-2bfe-4b03-ad97-685b3df9839d"> Prediction over validation set in red|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/8a259a57-1a45-4997-8a52-5c16f04f47f2"> A closer look at 4 intervals|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/452cea78-e06f-4eee-ab4e-a1ce96b09746"> Prediction over train + val, a closer look|
<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/96aaa31f-98ac-4a0e-91b8-4b859632d84f"> `(rate[0]/rate_err[0]-pred)`|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/ff0bf20d-3671-41f3-a7c5-a66c0fce95cb"> `(rate[0]/rate_err[0]-pred)` hist|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/b2355931-537f-4f9e-ba61-1ec0b5a29f16"> Losses (average mini-batch MSE loss)|

- And if we use our trained model with this "periodical split" dataset and apply it to the full dataset (train + val + test) including the 25 GRBs we removed, we can observe these:

| | |
|:-------------------------:|:-------------------------:|
|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/13f43dbe-9551-4a64-aeb5-1f08e9882fb5">|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/0d4b8857-fa19-4379-bf27-d79e5efe5d7e">|
<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/f8f22ef2-941f-4060-8246-307a3c66f089"> 38887 red dots|<img width="1604" src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/f613704c-d6c3-4d6e-98cd-11f4be0acda7">|

#### Investigating whether I showed the correct 25 GRBs

- Started investigating whether I showed the correct 25 GRBs and whether the conversion from UTC to `unix_time` wasn't wrong. This is because a few of my plots of the 25 GRBs are "flat" and look more like background than GRB... We can compare with GRBs from https://www.astro.unige.ch/polar/grb-light-curves?page=2 (for example with GRB 170114B)

- Checked manually through a few GRBs whether I showed wrong intervals, and it seemed that the `unix_time` conversion from UTC was correct (I even checked by downloading a few root files and compared their `unix_time`'s to what I obtained and they matched)

- Via this manual check, I discovered that the target might be different than what is shown on the website, I'm maybe training using the wrong targets.. where GRBs are sometimes not visible, therefore, detection based on residual thresholding wouldn't be successful for them.

- Actually, it might be also due to the binning, the website shows much much more precise light curves. The binning maybe caused some GRBs to lose in amplitude compared to the background ?

#### Documentation
- Started cleaning a bit the logbook
- Started writing one of the two report: Industrial report

### (Future) Goals:
- To better understand how to split the data into train, validation test set for our application as they are maybe some 'issues' related to overfitting when we shuffle our data and pick train, validation, and test set where examples can be close to each other in time (or other measurements). We maybe want to also take into account
temporal relationships. There's maybe something called "overfitting in feature space".

- Some links on splitting but our goal is not to forecast but to predict the "present" from the "present" (or maybe even past but not yet):
	- https://stats.stackexchange.com/questions/346907/splitting-time-series-data-into-train-test-validation-sets
	- https://datascience.stackexchange.com/questions/91162/why-is-shuffling-timeseries-a-bad-thing
- To read more about predicting a time series or sequence using multiple time series or sequences (something to explore) (and correlated residuals):
	- https://otexts.com/fpp2/regression.html
	- https://ethz.ch/content/dam/ethz/special-interest/math/statistics/sfs/Education/Advanced%20Studies%20in%20Applied%20Statistics/course-material-1921/Zeitreihen/ATSA_Script_v200504.pdf (from page 133)
- To better understand Adam optimizer, different parts of what I've used in general.
- To better understand or to learn more about Hydra
- To use W&B artifacts for datasets. Need to version datasets as I can work with different datasets
- To learn more about regularization, dropout, batch normalization
- To learn more about W&B sweeps and add more log information.
- To add a "stagnation end condition" to my training loop
- Is it fine to apply prediction over the whole dataset and threshold residuals to see whether known GRBs are part of them? (and what if we apply unsupervised learning outlier detection over the residuals ?)
- ACF of residuals, report, legends in my plots, feature importance, better threshold.

</details>

## Week 6: 07.08.23 - 13.08.23

<details>

### Summary

- Fixed CUDA out-of-memory issue (which happened with my bigger model because it had more parameters on GPU):
	- Instead of directly loading the whole dataset into GPU (even before `__getitem__` in the PyTorch Dataset), I load it into GPU
one by one inside the `__getitem__` method (creating tensors on CUDA for each example)
	- `torch.no_grad()` in `visualizer.py` (and `./notebooks/results.ipynb`) significantly decreased the allocated memory size. See: https://discuss.pytorch.org/t/how-to-delete-a-tensor-in-gpu-to-free-up-memory/48879/15
- However, creating a tensor on GPU for each example and every time we call `__getitem__` led to $+3$ times worse training execution time (from 3 hours ETA to 10 hours ETA).
- To fix the 10 hours ETA, I no longer create PyTorch tensors on GPU and in `__getitem__` but create them on CPU and at initialization of the PyTorch Dataset. A subset is moved to GPU when needed (e.g. before a mini-batch is fed to the model). It's now taking 1 hour 40 min instead of 3 or 10 hours to train the model.

- Checking again [GRB 170219A](https://www.astro.unige.ch/polar/content/170219a). There's something wrong with the spikes and the GRBs in our dataset:
![image](https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/2a7e2fa8-e9b6-4780-86b7-423c15dc822e)
![image](https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/0998659b-2861-4a83-9a7d-85acab370bde)

- Added custom loss (weighted MSE where weights are like `1/rate_err**2`) (it takes longer to train our model due to the parts where we retrieve the weights,
the `1/rate_err**2`, 
- `__getitem__` now also returns the idx, therefore, I could obtain the indices of the data from the mini-batch and use them to retrieve the correct weights as said previously.
- Added multiple targets (not just `rate[0]`) (did not try if old config yaml still works..)
- Tried to change `find_moments` in `./src/visualizer.py` to deal with 2D arrays instead of 1D arrays but had some complications, so I went back to how it was before the change. I use loops to go through the different targets and independently obtain the `new_std`...
- Tried to clean the code and tried to not use the `data_df` Pandas DataFrame inside `PolarDataset` but couldn't completely. See next.
- Discovered that the `unix_time` feature, which was in double precision (in `.root` format), was quantized when stored inside a PyTorch Tensor. This is because the default precision of a PyTorch FloatTensor is single precision, 32 bits. It implies that some features/inputs maybe lost some precision when going from the Pandas DataFrame to the PyTorch Tensor.
- I therefore, kept the `data_df` Pandas DataFrame attribute in order to retrieve the `unix_time` when visualizing the results but I must remember about the single vs double precision as it can affect the predictions.
- I did not change the PyTorch tensors datatypes to float64 because of memory usage and training speed.
- I added `num_workers=4` and `pin_memory` in the arguments of my PyTorch DataLoaders (thanks to Eloi Alonso) and my training time went from about 2 hours 15 min to 1 hour 30 min.
- Tried training with mini-batch sizes of $256$ and `periodicity` $256$ (reminder: it was in `periodical` split type).

We show in orange the prediction and in blue the $rate[0]$. In red, we have our points above the threshold of 3 std (again, as a reminder, it's actually `new_std` based on a modified gaussian fit).

![image](https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/08ef8d27-ee8b-4c5e-91dc-e222c131f7b4)

Residuals/error rate:

![image](https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/ad2e1508-b8d2-4096-8512-619053ac1c94)

Our model also predicted other rates but we don't show them yet as there would be too many plots. We got:
```
Number of red points when thresholding using residuals/rate_errs (pull)
[39175 37358 74967 73827 64414 64942]

>>> new_std
array([1.3854719, 1.2320681, 1.3846753, 1.313497 , 1.5823889, 1.5771829],
      dtype=float32)
```

### Some interesting links:

- https://medium.com/syncedreview/how-to-train-a-very-large-and-deep-model-on-one-gpu-7b7edfe2d072
- https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-asynchronous-data-loading-and-augmentation
- https://androidkt.com/pytorch-dataloader-set-pin_memory-to-true/
- https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234

### (Future) goals:
- Group the data above threshold (my red points) into groups/clusters (instead of looking at them individually, point by point)
- Clean code (especially `src/visualizer.py` and `notebooks/results.ipynb`), clean logebook, fix documentation with the config file
- Check that old config file still work with current code (e.g. when we had only one target and used MSELoss as criterion).

</details>


## Week 7: 14.08.23 - 20.08.23

<details>

### Summary
- Group the data above threshold (my red points) into groups/clusters (instead of looking at them individually, point by point). 
We show below, using a point for each cluster, the cluster length versus the cluster "integral" (sum of `rate[0]` for data in the cluster) where the threshold was set at $3$ std:

![image](https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/1746aa7c-1dd5-4dd4-8792-ebaa26515d3b)

There are still $15068$ clusters but it's still better than looking at $39175$ points.

- Ran training with old config file in which the target was just `rate[0]/rate_err[0]` and the criterion with MSELoss (not weighted MSE Loss).
It gave very similar results but they were still different (it might be due to the fact that we want to predict smaller or higher values)
- Ran training with different `weight_decay` values from the Adam optimizer (L2 regularization). They seem to not help reduce the variance (reduce overfitting). I did not try (inverted) dropout yet. I don't want to try early stopping nor data augmentation. Note that I already normalized the inputs/features. I also did not try to tweak weight initialization (with for example Xavier or He initialization that are good with tanh and ReLU activation functions respectively, see [deep learning specialization on Coursera](https://www.coursera.org/learn/deep-neural-network/home/week/1).)

![W B Chart 8_16_2023, 2 22 05 PM](https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/c1a82cf4-e6a9-4f1f-a50f-3cedde1bc46c)

With run names (or ids):
![W B Chart 8_16_2023, 2 22 05 PM(1)](https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/15d56c35-f84f-4bba-98ef-14e21954d9b3)



- Learned more about Adam optimizer: it's a combination of RMSProp and Momentum (and includes bias correction). Essentially, it's doing two exponentially weighted moving averages, one on the gradients ("momentum" part) and one on the element-wise squared gradients ("RMSProp" part).
Intuitively, we can think of the "RMSProp" part as estimating the "variance" of the gradients and scale the steps in different directions accordingly.

- Learned that random sampling (not necessarily uniform, it could be uniform in log scale though, for instance for the beta in Momentum or alpha for the learning rate) when trying different hyperparamters can be useful.

- Didn't try batch normalization, it might be something to consider trying.

- Learned that what we're trying to do by plotting the residuals and where the predictions fail can be called "manual error analysis" but Andrew Ng only talks about classification tasks in which we can say whether we misclassified or not. What is usually done in regression ?
- Learned that I might need a second metric that captures what we want to do with the clusters or red points because the weighted MSE might not be enough.
- Learned that transfer learning cannot be used for tabular data (it is much more used in Computer Vision). Transfer learning is when we take a pretrained network, reuse the weights (can freeze them or not) and retrain a part of the network (either from the existing structure or new layers). Transfer learning is particularly useful when the task on which the pretrained network was trained has some similarities with the downstream task. Moreover, the upstream task should have used more data than the downstream task and the inputs for both tasks should be similar.
- Learned that what we're doing could be called "multi-task learning" as we can try to predict multiple output values (different rates).

</details>


## Week 8: 21.08.23 - 27.08.23

<details>

### Summary
- Ran training with a different split percentage: 98, 1, 1 (train, val, test) and it gave much much better validation loss as expected
(more training data, it can "reduce" overfitting). However, I still doubt about it.
- Ran again the training with the old small MLP with 3 hidden layers. Although the losses decrease more slowly than the bigger network, after a few epochs, they are very similar.
- Even though both losses are similar, clusters can change widely... it indicates us that we need another metric to track properties of these clusters.
- Learned a bit some basics about convolutional neural networks (because they can be used on time series):
	- convolution operations in CNNs are actually cross-correlations. Implicit ReLU after applying convolution
	- convolution over volumes (dimension >= 2D tensors)
  	- padding, striding
   	- pooling
   	- 1 x 1 convolution to reduce the depth/number of channel but keep the height and width
    	- some classical CNN architectures: VGG, AlexNet, LeNet-5, GoogleNet (or Inception network)
- Learned briefly about ResNets and residual blocks with skip connections. It helps with exploding, vanishing gradients in deep networks.
it might be useful if we want to create a very deep network for our problem.
- Cleaned a bit the code, especially the part where we can use different loss functions in `../src/trainer.py`.
- Completely removed old jupyter notebooks and src files in `../notebooks`.
- Modified `../notebooks/README.md` with some information on how to get to these old notebooks.
- Updated documentation with a new description of the notebooks folder and some description how to specify the loss function from `../config/trainer.yaml`
- Cleaning `../notebooks/results.ipynb` and added interactive plots using ipywidgets. However, there was an issue with file size going from 1 mb to 1 gb...
- Learned very very quickly about transforming FC (fully connected) layers into convolutional layers (some using 1 x 1 convolutions).
- Learned that one forward pass of CNN is "equivalent" but faster than doing multiple forward passes of smaller windows (see convolutional implementation of sliding windows). This might be useful if we want to use a CNN for time series ?
- Made my plots interactice using ipywidgets and ipympl. It's still to slow to update.
- Changed orange colored curves (prediction) into black colored curves.

### Some interesting links

- https://proclusacademy.com/blog/robust-scaler-outliers/


### TODO
- Permutation importance (be careful about colinearity or multi colinearity)
- Maybe robust scaler instead of standard scaler for normalizing inputs
- Interactive way to go through the clusters obtained from the red points
- Create clusters for points in which the prediction is higher than the target (we did the opposite until now)
- Find a single number evaluation metric that can be useful for our clusters in order to compare different models
- Make it easier to compare different models. Maybe use weights and biases for that
- Find how we can use CNN for time series ?
- Learn more about sequential models such as Transformers ?
- Find a better way to threshold and get red points ? It seems, for the moment, that it's not the most promising directions. The predictions or our red points can change widely just because some residuals can be higher or lower for some unknown reasons

</details>

## Week 9: 28.08.23 - 03.09.23

<details>

### Summary
- Made my plots even more interactice using ipywidgets and ipympl and made them faster to update/refresh.
- We can click on the different "dots" on the left to select a cluster and it will automatically show an interval around that cluster. Red points are points from that cluster. As before, the blue curve represents the original curve and the black curve represents the prediction. The lower right plot represents the residuals or residuals/error rates.

- Selecting a particular cluster will highlight it in black on the left plot as well as change the right plot's title. Green vertical lines in the right plot represents a known GRB trigger time (out of the 25) (Note: they're not always visible depending on which window we're looking at).
We can also zoom in the plots and move around.

- Here is a screenshot of the sliders where:
	- `w`: window size
	- `k`: used for the threshold $k\cdot \sigma$
	- `pred_below`: 1 for data s.t. residuals or pulls $> k\cdot \sigma$ and 0 for $< -k\cdot \sigma$

![image](https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/70ece583-8c6e-4a07-aae3-2a6c5b9becc1)

- Some examples without the sliders/buttons where pulls $> k\cdot \sigma$

| <img src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/a3536bbf-c8a9-4e88-8d52-ffdc4deac9de" width=300> | <img src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/d505e940-009f-4ab2-aee8-b418f9f37bc2" width=300>| <img src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/f4653965-bd63-47b6-a103-6a6153117adb" width=300>|
|:--:|:--:|:--:|

- Some examples without the sliders/buttons where pulls $< -k\cdot \sigma$
 
| <img src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/36684a50-6757-4d06-99e9-7a53fe058253" width=300> | <img src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/9375257a-4f7d-4a19-86ff-b871d8174f14" width=300>| <img src="https://github.com/Zenchiyu/POLAR-background-prediction/assets/49496107/2e7521e3-8d6b-4758-b8ba-bcb40f4d2070" width=300>|
|:--:|:--:|:--:|

- Fixed `integral`, it wasn't summing using the correct target.
- Added `normalized pull` = `pull/new_std` but didn't add it in every plot. The threshold stays the same as before.
- Fixed errors with `len()` in the scatter plots of the interactive plot that used mpl-interactions.
- Fixed dataframe index of `data_df`.
- Fixing issue with red data points legend not showing up if first axis doesn't have one red data point
- Tried to create cluster intersections with Nicolas Produit using for loops through some variable-sized array.
- Abandonned that code and used pandas dataframe merge + some other operations to get the cluster intersections.
- Found that the merge didn't keep the unix time order nor the index from `data_df`.
- Fixed it by setting the sort argument to True and putting back the correct index after the merge.  
- Fixed `discard_w`. There was an issue with the ending points.
- Cleaning a part of my code

### Some interesting links

- https://proclusacademy.com/blog/robust-scaler-outliers/


### TODO
- Permutation importance (be careful about colinearity or multi colinearity)
- Maybe robust scaler instead of standard scaler for normalizing inputs
- ~~Interactive way to go through the clusters obtained from the red points~~ (Done)
- ~~Create clusters for points in which the prediction is higher than the target (we did the opposite until now)~~ (Done, see `pred_below`)
- Find a single number evaluation metric that can be useful for our clusters in order to compare different models
- Make it easier to compare different models. Maybe use weights and biases for that
- Find how we can use CNN for time series ?
- Learn more about sequential models such as Transformers ?
- Find a better way to threshold and get red points ? It seems, for the moment, that it's not the most promising directions. The predictions
or our red points can change widely just because some residuals can be higher or lower for some unknown
</details>

## Week 10: 04.09.23 - 10.09.23

<details>

### Summary

- Cleaning a part of my codes
- Tried to "interactively" save my interactive plots in PDF format to get selectable text from our figures. Couldn't find a way to do it so I went back to my first idea of printing the cluster information from the title. 
- Can now "interactively" print cluster information below my interactive plots. By using IPython.display `clear_output`, I cannot erase previous prints without erasing my plots.
- Changed `inter_id` to `inter_id_or_cond`: we can now show all clusters except those that don't appear in enough number of energy bins.

### TODO
- Feature importance, explain the weights, why the rate goes up or down. If can kill some weights, reduce the model complexity. Find the underlying rules that the model found.
- Clean logbook, clean code
- Start the two reports

</details>


## Week 11: 11.09.23 - 17.09.23

<details>

### Summary

</details>


## Week 12: 18.09.23 - 24.09.23

<details>

### Summary

</details>
