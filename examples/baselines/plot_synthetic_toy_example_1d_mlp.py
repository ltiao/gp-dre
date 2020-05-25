# -*- coding: utf-8 -*-
"""
MLP on Synthetic 1D Toy Problem
===============================
"""
# sphinx_gallery_thumbnail_number = 6

import numpy as np
import tensorflow.keras.backend as K
import tensorflow_probability as tfp

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from gpdre.base import DensityRatioMarginals, MLPDensityRatioEstimator
from gpdre.datasets import make_classification_dataset

K.set_floatx("float64")
# %%

# shortcuts
tfd = tfp.distributions

# constants
num_train = 2000  # nbr training points in synthetic dataset
num_features = 1  # dimensionality
num_index_points = 512  # nbr of index points

num_seeds = 3
num_layers_iter = range(1, 4)
num_units_log2_iter = range(3, 6)

optimizer = "adam"
epochs = 100
batch_size = 64

dataset_seed = 8888  # set random seed for reproducibility

x_min, x_max = -5.0, 5.0

# index points
X_grid = np.linspace(x_min, x_max, num_index_points).reshape(-1, num_features)

# %%
# Synthetic dataset
# -----------------

p = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(probs=[0.3, 0.7]),
    components_distribution=tfd.Normal(loc=[2.0, -3.0], scale=[1.0, 0.5]))
q = tfd.Normal(loc=0.0, scale=2.0)
r = DensityRatioMarginals(top=p, bot=q)

# %%
# Probability densities

fig, ax = plt.subplots()

ax.plot(X_grid, r.top.prob(X_grid), label='$q(x)$')
ax.plot(X_grid, r.bot.prob(X_grid), label='$p(x)$')

ax.set_xlabel('$x$')
ax.set_ylabel('density')

ax.legend()

plt.show()

# %%
# Log density ratio, log-odds, or logits.

fig, ax = plt.subplots()

ax.plot(X_grid, r.logit(X_grid), c='k',
        label=r"$f(x) = \log p(x) - \log q(x)$")

ax.set_xlabel('$x$')
ax.set_ylabel('$f(x)$')

ax.legend()

plt.show()

# %%
# Density ratio.

fig, ax = plt.subplots()

ax.plot(X_grid, r.ratio(X_grid), c='k', label=r"$r(x) = \exp{f(x)}$")

ax.set_xlabel('$x$')
ax.set_ylabel('$r(x)$')

ax.legend()

plt.show()

# %%
# Create classification dataset.

X_p, X_q = r.make_dataset(num_train, seed=dataset_seed)
X_train, y_train = make_classification_dataset(X_p, X_q)

# %%
# Dataset visualized against the Bayes optimal classifier.

fig, ax = plt.subplots()

ax.plot(X_grid, r.prob(X_grid), c='k',
        label=r"$\pi(x) = \sigma(f(x))$")
ax.scatter(X_train, y_train, c=y_train, s=12.**2,
           marker='s', alpha=0.1, cmap="coolwarm_r")
ax.set_yticks([0, 1])
ax.set_yticklabels([r"$x_q \sim q(x)$", r"$x_p \sim p(x)$"])
ax.set_xlabel('$x$')

ax.legend()

plt.show()
# %%

rows = []

for num_layers in num_layers_iter:

    for num_units_log2 in num_units_log2_iter:

        num_units = 1 << num_units_log2

        for seed in range(num_seeds):

            r_mlp = MLPDensityRatioEstimator(num_layers=num_layers,
                                             num_units=num_units,
                                             activation="relu", seed=seed)
            r_mlp.compile(optimizer=optimizer, metrics=["accuracy"])
            r_mlp.fit(X_p, X_q, epochs=epochs, batch_size=batch_size)

            logit = r_mlp.logit(X_grid)
            ratio = r_mlp.ratio(X_grid)

            rows.append(pd.DataFrame(dict(num_layers=num_layers,
                                          num_units=num_units,
                                          seed=seed,
                                          x=X_grid.squeeze(),
                                          logit=logit,
                                          ratio=ratio)))
# %%

data = pd.concat(rows, axis="index", sort=True)
data
# %%

fig, ax = plt.subplots()

ax.plot(X_grid, r.logit(X_grid), c='k', label=r"$r(x) = \exp{f(x)}$")
sns.lineplot(x="x", y="logit", hue="num_layers", size="num_units",
             units="seed", estimator=None, palette="colorblind",
             alpha=0.6, linewidth=1.0, data=data, ax=ax)

plt.show()
# %%

fig, ax = plt.subplots()

ax.plot(X_grid, r.ratio(X_grid), c='k', label=r"$r(x) = \exp{f(x)}$")
sns.lineplot(x="x", y="ratio", hue="num_layers", size="num_units",
             units="seed", estimator=None, palette="colorblind",
             alpha=0.6, linewidth=1.0, data=data, ax=ax)

plt.show()
