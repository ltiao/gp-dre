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
from gpdre.plotting import continuous_pairplot
# %%

# constants
data_home = "../../datasets/"

# %%

rows = []

for dataset_name in DATASET_LOADER.keys():

    train_data, test_data = load_dataset(dataset_name, data_home=data_home)

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
data

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
# Visualize the lower-dimensional datasets.

train_data, test_data = load_dataset("puma8nh", data_home=data_home)

g = continuous_pairplot(*train_data, plot_kws=dict(s=6.0, alpha=0.6))

# %%

train_data, test_data = load_dataset("kin8nm", data_home=data_home)

g = continuous_pairplot(*train_data, plot_kws=dict(s=6.0, alpha=0.6))

# %%

train_data, test_data = load_dataset("bank8fm", data_home=data_home)

g = continuous_pairplot(*train_data, plot_kws=dict(s=6.0, alpha=0.6))

# %%

train_data, test_data = load_dataset("cpu_small", data_home=data_home)

g = continuous_pairplot(*train_data, plot_kws=dict(s=6.0, alpha=0.6))

# %%

train_data, test_data = load_dataset("elevators", data_home=data_home)

g = continuous_pairplot(*train_data, plot_kws=dict(s=6.0, alpha=0.6))
