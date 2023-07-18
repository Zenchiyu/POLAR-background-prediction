# Logbook


## Week 1: 03.07.23 - 09.07.23

I transformed the dataset in the `.root` format into other formats which can be manipulated inside Python. I also preprocessed
the dataset in a certain way (not necessarily final). I visualized the dataset and what we wanted to predict.

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
it's not completely correct to use FFT. However, due to the [orbital period of Tiangong-2 space lab](https://en.wikipedia.org/wiki/Tiangong-2), Earth's rotation, etc., there are seasonalities involved.
we could still observe a spike around "per 1 hour 30" frequency in the magnitude spectrum of the light curve.

- Using the results of FFT, we want to perform some operations in Fourier domain before constructing the light curve and use it as a target in our regression problem:
	- Started to code something in order to kill the spikes in the magnitude spectrum using two methods:
		- Using a box filter on the magnitude (but what window size should we use ? what padding method ?)
		- Applied linear regression in the "log x, log y" magnitude plot as it looked like some power law then wanted to use it to find the spikes before killing only the spikes (unfinished as we moved on to another idea)
	- We stopped trying to kill the spikes and started to think about killing low frequencies instead as we're mostly interested in high frequencies
due to GRBs (Gamma Ray Bursts) which could cause visible spikes:
		- We manually chose some threshold based on the magnitude spectrum to kill some low frequencies as well as some spikes




### (Future) Goals:
- 

## Week 2: 10.07.23 - 16.07.23


## Week 3: 18.07.23 - 23.07.23
