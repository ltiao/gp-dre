"""Plotting module."""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import KBinsDiscretizer


def line(x, y, ax=None, *args, **kwargs):

    a = np.minimum(x.min(), y.min())
    b = np.maximum(x.max(), y.max())

    u = [a, b]

    if ax is None:
        ax = plt.gca()

    ax.plot(u, u, linestyle="--", c="tab:gray")


def continuous_pairplot(features, target, columns=None, n_bins=4,
                        palette="viridis", corner=True, *args, **kwargs):
    """
    Pairplot with discretized continuous hue values.
    """
    scaler = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    quartile = 1 + scaler.fit_transform(target.reshape(-1, 1)).squeeze()

    data = pd.DataFrame(features, columns=columns).assign(quartile=quartile)

    return sns.pairplot(data=data, hue="quartile", palette=palette,
                        corner=corner, *args, **kwargs)


def fill_between_stddev(X_pred, mean_pred, stddev_pred, n=1, ax=None, *args,
                        **kwargs):

    if ax is None:
        ax = plt.gca()

    ax.fill_between(X_pred,
                    mean_pred - n * stddev_pred,
                    mean_pred + n * stddev_pred, **kwargs)
