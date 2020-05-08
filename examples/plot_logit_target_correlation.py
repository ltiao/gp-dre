# -*- coding: utf-8 -*-
"""
Correlations between logits and targets
=======================================
"""
# sphinx_gallery_thumbnail_number = 3

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from gpdre.benchmarks import get_cortes_splits
from gpdre.applications.covariate_shift.benchmarks import regression_metric
from gpdre.plotting import line

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_boston

from operator import lt, gt

# %%

# constants
test_rate = 1/3
dataset_seed = 8888
# %%


def make_data(split, uniform, exact):

    data = pd.DataFrame(dict(split=split,
                             exact=exact,
                             uniform=uniform))
    data.index.name = "seed"
    data.reset_index(inplace=True)

    return data
# %%


def make_data_full(splits, metrics_uniform, metrics_exact):

    dfs = []

    for split, uniform, exact in zip(splits, metrics_uniform, metrics_exact):

        dfs.append(make_data(split, uniform, exact))

    return pd.concat(dfs, axis="index", ignore_index=True, sort=True)
# %%


def get_data_diff(data):

    d = data.set_index("split")

    diff_series = d["uniform"] - d["exact"]
    diff_series.name = "nmse_diff"
    return diff_series.reset_index()
# %%


def make_yy_data(X, y, splits, rs):

    target_scaler = StandardScaler()
    z = target_scaler.fit_transform(y.reshape(-1, 1)).squeeze(axis=-1)

    dfs = []

    for split, r in zip(splits, rs):

        dfs.append(pd.DataFrame(dict(split=split, y=z, logit=r.logit(X))))

    return pd.concat(dfs, axis="index", ignore_index=True, sort=True)
# %%


dataset = load_boston()

X, X_test, y, y_test = train_test_split(dataset.data,
                                        dataset.target,
                                        test_size=test_rate,
                                        random_state=dataset_seed)
# %%
# Positive Differences
# --------------------
#
# Splits that result in *positive* differences between methods trained with
# uniform and exact weights.


splits, rs, metrics_uniform, metrics_exact = get_cortes_splits(regression_metric,
                                                               train_data=(X, y),
                                                               test_data=(X_test, y_test),
                                                               num_splits=9,
                                                               max_iter=10000,
                                                               num_seeds=10,
                                                               tol=0.1,
                                                               compare_pred=gt)
# %%

data = make_data_full(splits, metrics_uniform, metrics_exact)
yy_data = make_yy_data(X, y, splits, rs)
diff = get_data_diff(data)

# %%

g = sns.relplot(x="uniform", y="exact", hue="split",
                kind="scatter", data=data, alpha=0.7, palette="Set1",
                facet_kws=dict(sharex="row", sharey="row"))
g.map(line, "uniform", "exact")
# %%

fig, ax = plt.subplots()

sns.stripplot(x="split", y="nmse_diff", data=diff, palette="colorblind",
              alpha=0.4, zorder=1, ax=ax)

sns.pointplot(x="split", y="nmse_diff",
              data=diff, palette="dark", join=False, ci=None, markers='d',
              scale=0.75, ax=ax)

plt.show()
# %%

g = sns.lmplot(x="logit", y="y", col="split", col_wrap=3, data=yy_data)
# %%
# Negative Differences
# --------------------
#
# Splits that result in *negative* differences between methods trained with
# uniform and exact weights.

splits, rs, metrics_uniform, metrics_exact = get_cortes_splits(regression_metric,
                                                               train_data=(X, y),
                                                               test_data=(X_test, y_test),
                                                               num_splits=9,
                                                               max_iter=10000,
                                                               num_seeds=10,
                                                               tol=-0.1,
                                                               compare_pred=lt)
# %%

data = make_data_full(splits, metrics_uniform, metrics_exact)
yy_data = make_yy_data(X, y, splits, rs)
diff = get_data_diff(data)

# %%

g = sns.relplot(x="uniform", y="exact", hue="split",
                kind="scatter", data=data, alpha=0.7, palette="Set1",
                facet_kws=dict(sharex="row", sharey="row"))
g.map(line, "uniform", "exact")
# %%

fig, ax = plt.subplots()

sns.stripplot(x="split", y="nmse_diff", data=diff, palette="colorblind",
              alpha=0.4, zorder=1, ax=ax)

sns.pointplot(x="split", y="nmse_diff",
              data=diff, palette="dark", join=False, ci=None, markers='d',
              scale=0.75, ax=ax)

plt.show()
# %%

g = sns.lmplot(x="logit", y="y", col="split", col_wrap=3, data=yy_data)
