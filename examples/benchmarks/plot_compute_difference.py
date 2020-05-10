# -*- coding: utf-8 -*-
"""
Compute Largest Difference
==========================
"""
# sphinx_gallery_thumbnail_number = 1

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from gpdre.benchmarks import CortesDensityRatio
from gpdre.metrics import normalized_mean_squared_error

from sklearn.svm import SVR

from sklearn.datasets import load_diabetes, load_boston
from sklearn.preprocessing import MinMaxScaler

# %%

# constants
num_splits = 20
num_seeds = 10
scale = -2.0

model = SVR(kernel="rbf", gamma="scale", C=1.0, epsilon=0.1)
feature_scaler = MinMaxScaler()

# %%


def metric(model, X_train, y_train, X_test, y_test, sample_weight=None):

    model.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = model.predict(X_test)
    return normalized_mean_squared_error(y_test, y_pred)


def make_data(rows):

    return pd.DataFrame(rows).set_index(["split", "seed"])


def get_metrics_pair(X, y, num_splits=20, num_seeds=10, scale=-2.0):

    input_dim = X.shape[-1]

    rows_uniform = []
    rows_exact = []

    for split in range(num_splits):

        r = CortesDensityRatio(input_dim=input_dim, scale=scale, seed=split)

        for seed in range(num_seeds):

            (X_train, y_train), (X_test, y_test) = r.train_test_split(X, y, seed=seed)

            nmse = metric(model, X_train, y_train, X_test, y_test)
            rows_uniform.append(dict(split=split, seed=seed, nmse=nmse))

            nmse = metric(model, X_train, y_train, X_test, y_test,
                          sample_weight=r.ratio(X_train) + 1e-6)
            rows_exact.append(dict(split=split, seed=seed, nmse=nmse))

    data_uniform = make_data(rows_uniform)
    data_exact = make_data(rows_exact)

    return data_uniform, data_exact


def get_metrics_diff(data_uniform, data_exact):

    return data_uniform.sub(data_exact)


def get_split_largest_diff(data_diff):

    data_diff_mean = data_diff.groupby(level="split").mean()

    return data_diff_mean["nmse"].argmax()

# %%


dataset = load_boston()

X = feature_scaler.fit_transform(dataset.data)
y = dataset.target

# %%

data_uniform, data_exact = get_metrics_pair(X, y, num_splits=num_splits,
                                            num_seeds=num_seeds, scale=scale)
data = pd.concat([data_uniform.assign(importance="uniform"),
                  data_exact.assign(importance="exact")],
                 axis="index", sort=True).reset_index()

# %%

g = sns.catplot(x="importance", y="nmse", hue="split",
                data=data, kind="point", alpha=0.8, dodge=True,
                join=True, markers="d", scale=0.8, palette="tab20",
                facet_kws=dict(sharex=False, sharey=False))
# %%

data_diff = get_metrics_diff(data_uniform, data_exact)

# %%

fig, ax = plt.subplots()

sns.stripplot(x="split", y="nmse", data=data_diff.reset_index(),
              palette="tab20", alpha=0.4, zorder=1, ax=ax)

sns.pointplot(x="split", y="nmse", data=data_diff.reset_index(),
              palette="tab20", join=False, ci=None, markers='d',
              scale=0.75, ax=ax)

plt.show()

# %%

get_split_largest_diff(data_diff)
