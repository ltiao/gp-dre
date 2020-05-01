# -*- coding: utf-8 -*-
"""
LIACC Regression Datasets
=========================
"""
# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

from gpdre.datasets.liacc_regression import load_dataset, DATASET_LOADER
from sklearn.preprocessing import KBinsDiscretizer

# %%

rows = []

for dataset_name in DATASET_LOADER.keys():

    train_data, test_data = load_dataset(dataset_name)

    X_train, y_train = train_data

    num_train, num_features = X_train.shape

    num_test = 0

    if test_data is not None:

        X_test, y_test = test_data
        num_test, _ = X_test.shape

    rows.append(dict(dataset_name=dataset_name, num_train=num_train,
                     num_test=num_test, num_features=num_features))

# %%

data = pd.DataFrame(rows)

# %%

fig, ax = plt.subplots()

sns.scatterplot(x="num_train", y="num_features",
                size="num_test", hue="dataset_name",
                data=data, ax=ax)

plt.show()


# %%
# Copy-and-paste directly into a LaTeX document:

print(data.to_latex())

# %%
# Inspect one of the datasets.

(X, y), _ = load_dataset("puma8nh")

scaler = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
quartile = 1 + scaler.fit_transform(y.reshape(-1, 1)).squeeze()

puma8nh_data = pd.DataFrame(X).assign(quartile=quartile)

g = sns.pairplot(data=puma8nh_data, hue="quartile", palette="colorblind",
                 corner=True, plot_kws=dict(s=8.0, alpha=0.6))
