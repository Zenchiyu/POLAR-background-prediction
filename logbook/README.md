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


## Week 3: 18.07.23 - 23.07.23

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
- TODO: continue report, clean further the logbook, and readme. add unit tests?

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
