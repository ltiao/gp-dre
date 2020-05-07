# -*- coding: utf-8 -*-
"""
Label Shift
===========
"""
# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from gpdre.benchmarks import LabelShift
from gpdre.applications.covariate_shift.benchmarks import regression_metric
from gpdre.datasets.liacc_regression import DATASET_LOADER
from gpdre.plotting import continuous_pairplot, line

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_boston
# %%

# constants
test_rate = 1 / 3
dataset_seed = 8888
num_seeds = 10

DATASET_LOADER_SKLEARN = {
    "boston": load_boston,
    "diabetes": load_diabetes
}

dataset_names = []
dataset_names.extend(DATASET_LOADER_SKLEARN.keys())
dataset_names.extend(DATASET_LOADER.keys())
# %%


def make_data(dataset_name, exact, uniform):

    data = pd.DataFrame(dict(dataset_name=dataset_name,
                             exact=exact, uniform=uniform))
    data.index.name = "seed"
    data.reset_index(inplace=True)

    return data
# %%


dataset = load_diabetes()
r = LabelShift()
g = continuous_pairplot(features=dataset.data,
                        target=r.prob(dataset.data, dataset.target).numpy(),
                        columns=dataset.feature_names)
# %%

dfs = []

for dataset_name in dataset_names:

    if dataset_name in DATASET_LOADER_SKLEARN:

        dataset = DATASET_LOADER_SKLEARN[dataset_name]()
        X_all = dataset.data
        y_all = dataset.target

    else:

        (X_all, y_all), test_data = DATASET_LOADER[dataset_name]()

    X, X_test, y, y_test = train_test_split(X_all, y_all,
                                            test_size=test_rate,
                                            random_state=dataset_seed)

    r = LabelShift()

    metrics_uniform_seeds = []
    metrics_exact_seeds = []

    for seed in range(num_seeds):

        (X_train, y_train), (X_val, y_val) = r.train_test_split(X, y, seed=seed)

        metrics_uniform_seeds.append(regression_metric(X_train, y_train, X_test, y_test))
        metrics_exact_seeds.append(regression_metric(X_train, y_train, X_test, y_test,
                                                     sample_weight=np.maximum(1e-6, r.ratio(X_train, y_train))))

    dfs.append(make_data(dataset_name, metrics_exact_seeds, metrics_uniform_seeds))
# %%

data = pd.concat(dfs, axis="index", ignore_index=True, sort=True)
# %%

g = sns.relplot(x="uniform", y="exact", hue="dataset_name",
                kind="scatter", data=data, alpha=0.7, palette="tab10",
                facet_kws=dict(sharex="row", sharey="row"))
g.map(line, "uniform", "exact")
