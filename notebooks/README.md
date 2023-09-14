## Structure

This directory contains notebook(s) to explore/visualize the results when we use `nf1rate.root`.

```
.
├── formats
├── images
├── README.md
└── results.ipynb

3 directories, 2 files
```

The `formats` folder is used to store, in other formats such as PDF or HTML, some potentially old notebooks.

## Old structure

To see old notebooks as well as the old README, you can go to [the repository at an old commit](https://github.com/Zenchiyu/POLAR-background-prediction/tree/5bf385555594e0ad319f9a084e954795c56ad37c/notebooks) or git checkout to this commit: `5bf385555594e0ad319f9a084e954795c56ad37c`.

This directory contained different notebooks that we used to explore the data and create basic models (linear regression or sklearn MLPregressor).
Most of them are not needed for more complex models (are implemented in PyTorch in `src` folder and configured via configuration files such `trainer.yaml` in `config` directory)

```
.
├── dataset_f1rate.ipynb
├── dataset_fmrate.ipynb
├── dataset.ipynb
├── exploring_polar_data.ipynb
├── fmrate_prediction.ipynb
├── frequencies.py
├── linear_regression.ipynb
├── plotting.py
├── README.md
└── results.ipynb

0 directories, 10 files
```

Note: You can ignore `dataset_f1rate.ipynb` as we don't provide the dataset and we don't really use it afterwards. We keep it just for the `logbook`.

We wrote or ran the code in this order:
- `exploring_polar_data.ipynb`
- `dataset.ipynb`
- `linear_regression.ipynb`
- (`dataset_f1rate.ipynb`)
- `dataset_fmrate.ipynb`
- `fmrate_prediction.ipynb`
- `results.ipynb`

But the only two files you could run (as the dataset is in the `data` folder) were `dataset_fmrate.ipynb` and `fmrate_prediction`.

