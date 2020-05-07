# -*- coding: utf-8 -*-
"""
Cortes et al. 2008 Benchmark Set-up
===================================
"""
# sphinx_gallery_thumbnail_number = 1

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from gpdre.benchmarks import CortesDensityRatio
from gpdre.metrics import normalized_mean_squared_error
from gpdre.datasets.liacc_regression import load_dataset, DATASET_LOADER
from gpdre.plotting import line

from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# %%

# constants
num_samples = 2000

num_splits = 10
num_seeds = 5

scale = -4.0
test_rate = 1 / 3

dataset_seed = 8888

MODELS = {
    "kernel_ridge": lambda gamma: KernelRidge(kernel="rbf", gamma=gamma),
    "svr": lambda gamma: SVR(kernel="rbf", gamma=gamma, C=1.0, epsilon=0.1)
}
# %%


def metric(model, X_train, y_train, X_test, y_test, X_val, y_val,
           sample_weight=None):

    feature_scaler = StandardScaler()

    Z_train = feature_scaler.fit_transform(X_train)
    Z_test = feature_scaler.transform(X_test)
    Z_val = feature_scaler.transform(X_val)

    model.fit(Z_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(Z_test)
    y_pred_val = model.predict(Z_val)

    nmse_test = normalized_mean_squared_error(y_test, y_pred)
    nmse_val = normalized_mean_squared_error(y_val, y_pred_val)

    return nmse_test, nmse_val

# %%


def make_data(rows):

    return pd.DataFrame(rows).set_index(["dataset_name", "model_name",
                                         "gamma_name", "split", "seed",
                                         "partition"])
# %%


def get_metrics_diff(data_uniform, data_exact):

    return data_uniform.sub(data_exact)
# %%


def compute_gamma(X, foo="scale"):

    D = X.shape[-1]

    if foo == "scale":
        X_var = X.var()
        gamma = 1.0 / (D * X_var) if X_var != 0 else 1.0
    elif foo == "auto":
        gamma = 1.0 / D
    elif foo == "sqrt":
        gamma = np.sqrt(0.5 * D)
    else:
        raise ValueError

    return gamma
# %%


rows_uniform = []
rows_exact = []

# for dataset_name in list(DATASET_LOADER.keys())[:3]:
for dataset_name in DATASET_LOADER.keys():

    # `test_data` is ignored
    (X_all, y_all), test_data = load_dataset(dataset_name)

    X_subset, _, y_subset, _ = train_test_split(X_all, y_all,
                                                train_size=num_samples,
                                                random_state=dataset_seed)

    X, X_test, y, y_test = train_test_split(X_subset, y_subset,
                                            test_size=test_rate,
                                            random_state=dataset_seed)

    input_dim = X.shape[-1]

    for split in range(num_splits):

        r = CortesDensityRatio(input_dim=input_dim, scale=scale, seed=split)

        for seed in range(num_seeds):

            (X_train, y_train), (X_val, y_val) = r.train_test_split(X, y, seed=seed)

            for model_name in MODELS.keys():

                for gamma_name in ["scale", "auto", "sqrt"]:

                    gamma = compute_gamma(X_train, gamma_name)

                    model = MODELS[model_name](gamma)
                    nmse_test, nmse_val = metric(model, X_train, y_train,
                                                 X_test, y_test, X_val, y_val)
                    rows_uniform.append(dict(dataset_name=dataset_name,
                                             model_name=model_name,
                                             gamma_name=gamma_name,
                                             split=split, seed=seed,
                                             nmse=nmse_test, partition="test"))
                    rows_uniform.append(dict(dataset_name=dataset_name,
                                             model_name=model_name,
                                             gamma_name=gamma_name,
                                             split=split, seed=seed,
                                             nmse=nmse_val, partition="val"))

                    model = MODELS[model_name](gamma)
                    nmse_test, nmse_val = metric(model, X_train, y_train,
                                                 X_test, y_test, X_val, y_val,
                                                 sample_weight=np.maximum(1e-6, r.ratio(X_train)))
                    rows_exact.append(dict(dataset_name=dataset_name,
                                           model_name=model_name,
                                           gamma_name=gamma_name,
                                           split=split, seed=seed,
                                           nmse=nmse_test, partition="test"))
                    rows_exact.append(dict(dataset_name=dataset_name,
                                           model_name=model_name,
                                           gamma_name=gamma_name,
                                           split=split, seed=seed,
                                           nmse=nmse_val, partition="val"))
# %%

data_uniform = make_data(rows_uniform)
data_exact = make_data(rows_exact)

# %%

data = pd.concat([data_uniform.assign(importance="uniform"),
                  data_exact.assign(importance="exact")],
                 axis="index", sort=True).reset_index()

# %%

pivot_data = pd.pivot_table(data, columns="importance", values="nmse",
                            index=["dataset_name", "model_name", "gamma_name",
                                   "split", "seed", "partition"]).reset_index()

# %%
# Test set
# --------

g = sns.relplot(x="uniform", y="exact", hue="split",  style="gamma_name",
                row="dataset_name", col="model_name", kind="scatter",
                data=pivot_data[pivot_data.partition == "test"],
                alpha=0.7, palette="tab10",
                facet_kws=dict(sharex="row", sharey="row"))
g.map(line, "uniform", "exact")

# %%
# Validation set
# --------------

g = sns.relplot(x="uniform", y="exact", hue="split",  style="gamma_name",
                row="dataset_name", col="model_name", kind="scatter",
                data=pivot_data[pivot_data.partition == "val"],
                alpha=0.7, palette="tab10",
                facet_kws=dict(sharex="row", sharey="row"))
g.map(line, "uniform", "exact")

# %%
# Gamma auto
# ----------

g = sns.relplot(x="uniform", y="exact", hue="split",  style="partition",
                row="dataset_name", col="model_name", kind="scatter",
                data=pivot_data[pivot_data.gamma_name == "auto"],
                alpha=0.7, palette="tab10",
                facet_kws=dict(sharex="row", sharey="row"))
g.map(line, "uniform", "exact")

# %%

data_diff = get_metrics_diff(data_uniform, data_exact).reset_index()

# %%


def a(split, *args, **kwargs):

    ax = plt.gca()
    ax.plot(split, np.zeros_like(split), linestyle="--", c="tab:gray")

# %%
# Test set
# --------


g = sns.catplot(x="split", y="nmse", hue="gamma_name",
                row="dataset_name", col="model_name",
                palette="colorblind", alpha=0.6,
                data=data_diff[data_diff.partition == "test"])
g.map(a, "split")
g.set(ylim=(-0.75, 0.75))

# %%
# Validation set

g = sns.catplot(x="split", y="nmse", hue="gamma_name",
                row="dataset_name", col="model_name",
                palette="colorblind", alpha=0.6,
                data=data_diff[data_diff.partition == "val"])
g.map(a, "split")
g.set(ylim=(-0.75, 0.75))

# %%
# Gamma auto
# ----------

g = sns.catplot(x="split", y="nmse", hue="partition",
                row="dataset_name", col="model_name",
                palette="colorblind", alpha=0.6,
                data=data_diff[data_diff.gamma_name == "auto"])
g.map(a, "split")
g.set(ylim=(-0.75, 0.75))
