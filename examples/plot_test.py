# -*- coding: utf-8 -*-
"""
Schemes
=======
"""
# sphinx_gallery_thumbnail_number = 1

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from gpdre.evaluation import CortesDensityRatio, SugiyamaDensityRatio
from gpdre.metrics import normalized_mean_squared_error

from sklearn.svm import LinearSVR, SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

from sklearn.datasets import load_diabetes, load_boston
from sklearn.preprocessing import MinMaxScaler

# %%

# constants
num_splits = 9
num_seeds = 10

DATASET_LOADER = {
    "boston": load_boston,
    "diabetes": load_diabetes
}

MODELS = {
    "linear_ridge": Ridge(alpha=5.0),
    "kernel_ridge": KernelRidge(kernel="rbf"),
    # "linear_svr": LinearSVR(max_iter=12000),
    "svr": SVR(kernel="rbf", gamma="scale", C=1.0, epsilon=0.1)
}

# %%


def line(x, y, *args, **kwargs):

    # a = np.minimum(x.min(), y.min())
    # b = np.maximum(x.max(), y.max())
    a = 0.2
    b = 2.0

    u = [a, b]

    ax = plt.gca()

    ax.plot(u, u, linestyle="--", c="tab:gray")


def metric(model, X_train, y_train, X_test, y_test, sample_weight=None):

    model.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = model.predict(X_test)
    return normalized_mean_squared_error(y_test, y_pred)

# %%


rows = []

for dataset_name, load_dataset in DATASET_LOADER.items():

    dataset = load_dataset()

    feature_scaler = MinMaxScaler()
    # target_scaler = StandardScaler()

    X = feature_scaler.fit_transform(dataset.data)
    # X = dataset.data
    y = dataset.target

    input_dim = X.shape[-1]

    for i in range(-1, 3):

        scale = - 2**i

        for split in range(num_splits):

            r = CortesDensityRatio(input_dim=input_dim, scale=scale, seed=split)

            for seed in range(num_seeds):

                (X_train, y_train), (X_test, y_test) = r.train_test_split(X, y, seed=seed)

                train_rate = len(X_train) / len(X)

                for model_name, model in MODELS.items():

                    nmse = metric(model, X_train, y_train, X_test, y_test)
                    rows.append(dict(dataset_name=dataset_name,
                                     model_name=model_name,
                                     importance="uniform", scale=scale,
                                     split=split, seed=seed,
                                     train_rate=train_rate, nmse=nmse))

                    nmse = metric(model, X_train, y_train, X_test, y_test,
                                  sample_weight=r.ratio(X_train) + 1e-6)
                    rows.append(dict(dataset_name=dataset_name,
                                     model_name=model_name,
                                     importance="exact", scale=scale,
                                     split=split, seed=seed,
                                     train_rate=train_rate, nmse=nmse))
# %%

data = pd.DataFrame(rows)

# %%

g = sns.catplot(x="split", y="nmse", hue="importance", style="dataset_name",
                row="model_name", col="scale", dodge=True,
                data=data[data.dataset_name == "boston"], kind="point",
                alpha=0.8, join=True, markers="d", scale=0.8,
                facet_kws=dict(sharex=False, sharey=False))

# %%

g = sns.catplot(x="split", y="nmse", hue="importance", style="dataset_name",
                row="model_name", col="scale", dodge=True,
                data=data[data.dataset_name == "diabetes"], kind="point",
                alpha=0.8, join=True, markers="d", scale=0.8,
                facet_kws=dict(sharex=False, sharey=False))

# %%

pivot_data = pd.pivot_table(data, columns="importance", values="nmse",
                            index=["dataset_name", "model_name", "scale",
                                   "split", "seed", "train_rate"]).reset_index()

# %%

g = sns.relplot(x="uniform", y="exact", hue="split", style="dataset_name",
                size="train_rate", row="model_name", col="scale",
                kind="scatter", data=pivot_data, alpha=0.7, palette="Set1",
                facet_kws=dict(sharex=False, sharey=False))
g.map(line, "uniform", "exact")

# %%

rows = []

for dataset_name, load_dataset in DATASET_LOADER.items():

    dataset = load_dataset()

    feature_scaler = MinMaxScaler()
    # target_scaler = StandardScaler()

    X = feature_scaler.fit_transform(dataset.data)
    # X = dataset.data
    y = dataset.target

    input_dim = X.shape[-1]

    for feature in range(input_dim):

        r = SugiyamaDensityRatio(feature=feature)

        for seed in range(num_seeds):

            (X_train, y_train), (X_test, y_test) = r.train_test_split(X, y, seed=seed)

            train_rate = len(X_train) / len(X)

            for model_name, model in MODELS.items():

                nmse = metric(model, X_train, y_train, X_test, y_test)
                rows.append(dict(dataset_name=dataset_name,
                                 model_name=model_name,
                                 importance="uniform", feature=feature,
                                 seed=seed, train_rate=train_rate, nmse=nmse))

                nmse = metric(model, X_train, y_train, X_test, y_test,
                              sample_weight=r.ratio(X_train) + 1e-6)
                rows.append(dict(dataset_name=dataset_name,
                                 model_name=model_name,
                                 importance="exact", feature=feature,
                                 seed=seed, train_rate=train_rate, nmse=nmse))

# %%

data = pd.DataFrame(rows)
pivot_data = pd.pivot_table(data, columns="importance", values="nmse",
                            index=["dataset_name", "model_name", "feature",
                                   "seed", "train_rate"]).reset_index()

# %%

g = sns.catplot(x="feature", y="nmse", hue="importance",
                row="model_name", col="dataset_name", dodge=True,
                data=data, kind="point", alpha=0.8,
                join=True, markers="d", scale=0.8,
                facet_kws=dict(sharex=False, sharey=False))

# %%

g = sns.relplot(x="uniform", y="exact", hue="feature",
                size="train_rate", row="model_name", col="dataset_name",
                kind="scatter", data=pivot_data, alpha=0.7, palette="tab20",
                facet_kws=dict(sharex=False, sharey=False))
g.map(line, "uniform", "exact")
