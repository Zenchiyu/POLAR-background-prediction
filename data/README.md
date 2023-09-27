# Data

This directory contains inputs or outputs to our methodology for extracting time intervals.

- `GRBs.csv` contains $55$ known GRB trigger times extracted from the paper [Overview of the GRB observation by POLAR](https://www.researchgate.net/profile/Yuanhao-Wang-8/publication/326811280_Overview_of_the_GRB_observation_by_POLAR/links/5cfe12c0a6fdccd1308f8b32/Overview-of-the-GRB-observation-by-POLAR.pdf)
- `cluster_inter_nf1rate.root` is a ROOT-CERN file containing start and end times of positive/negative cluster intersections (for different sets of energy bins or conditions) (initial seed $42$).
- `cluster_inter_nf1rate_<id>.root` where `<id>` is a run id, is a ROOT-CERN file containing the same as before but for a model trained on another seed ($10$ different seeds so $10$ different run ids).
- `pred_nf1rate.root` is a ROOT-CERN file containing the predictions (predicted photon rates for different energy bins) (initial seed $42$).
- `num_cluster_intersections.pkl` is a pickled Pandas DataFrame (can load it using `pd.read_pickle`) containing some properties of positive/negative cluster intersections (initial seed $42$).
- `num_cluster_intersections_<id>.pkl` where `<id>` is a run id, is a pickled Pandas DataFrame containing the same as before but for a model trained on another seed ($10$ different seeds so $10$ different run ids).
- `num_examples_clusters.pkl` is a pickled Pandas DataFrame (can load it using `pd.read_pickle`) containing some properties of positive/negative examples of interest or clusters (initial seed $42$).
- `num_examples_clusters_<id>.pkl` where `<id>` is a run id, is a pickled Pandas DataFrame containing the same as before but for the a model trained on another seed ($10$ different seeds so $10$ different run ids).
- `rate.txt` contains some information on energy bins for different target photon rates.

The dataset (e.g. `nf1rate.root`) should be placed under this directory.
