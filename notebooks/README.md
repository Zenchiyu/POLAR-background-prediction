## Structure

This directory contains different notebooks that we used to explore the data and create basic models (linear regression or sklearn MLPregressor).
They are not needed for more complex models (are implemented in PyTorch in `src` folder and configured via configuration files such `trainer.yaml` in `config` directory)

```
.
├── dataset_f1rate.ipynb
├── dataset_fmrate.ipynb
├── dataset.ipynb
├── exploring_polar_data.ipynb
├── fmrate_prediction.ipynb
├── linear_regression.ipynb
└── README.md

0 directories, 7 files
```

Note: You can ignore `dataset_f1rate.ipynb` as we don't provide the dataset and we don't really use it afterwards. We keep it just for the `logbook`.

We wrote or ran the code in this order:
- `exploring_polar_data.ipynb`
- `dataset.ipynb`
- `linear_regression.ipynb`
- (`dataset_f1rate.ipynb`)
- `dataset_fmrate.ipynb`
- `fmrate_prediction.ipynb`

But the only two files you can run (as the dataset is in the `data` folder) are `dataset_fmrate.ipynb` and `fmrate_prediction`.

