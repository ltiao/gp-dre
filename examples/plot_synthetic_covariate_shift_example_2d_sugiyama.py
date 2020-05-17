# -*- coding: utf-8 -*-
"""
Synthetic 2D Classification Covariate Shift Problem (Sugiyama et al. 2007)
==========================================================================
"""
# sphinx_gallery_thumbnail_number = 8

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

import matplotlib.pyplot as plt

from gpdre.benchmarks import SugiyamaKrauledatMuellerDensityRatioMarginals
from gpdre.datasets import make_classification_dataset

from sklearn.linear_model import LogisticRegression

# %%

# shortcuts
tfd = tfp.distributions
kernels = tfp.math.psd_kernels

# constants
num_train = 500
num_test = 500

num_layers = 2
num_units = 32
activation = "relu"
l2_factor = 5e-3

optimizer = "adam"
epochs = 400
batch_size = 32

C = 5.0

seed = 42
dataset_seed = 24

X2, X1 = np.mgrid[-4:9:200j, -6:8:200j]
X_grid = np.dstack((X1, X2))
# %%


def class_posterior(x1, x2):
    return 0.5 * (1 + tf.tanh(x1 - tf.nn.relu(-x2)))
# %%


r = SugiyamaKrauledatMuellerDensityRatioMarginals()
(X_train, y_train), (X_test, y_test) = r.make_covariate_shift_dataset(
  num_test, num_train, class_posterior_fn=class_posterior, seed=dataset_seed)
X, s = make_classification_dataset(X_test, X_train)
# %%

fig, ax = plt.subplots()

ax.set_title(r"Train $p_{\mathrm{tr}}(\mathbf{x})$ and "
             r"test $p_{\mathrm{te}}(\mathbf{x})$ distributions")

contours_train = ax.contour(X1, X2, r.bot.prob(X_grid), cmap="Blues")
contours_test = ax.contour(X1, X2, r.top.prob(X_grid), cmap="Oranges",
                           linestyles="--")

ax.clabel(contours_train, fmt="%.2f")
ax.clabel(contours_test, fmt="%.2f")

ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')

plt.show()

# %%

fig, ax = plt.subplots()

ax.set_title(r"$f(\mathbf{x}) = "
             r"\log p_{\mathrm{te}}(\mathbf{x}) - "
             r"\log p_{\mathrm{tr}}(\mathbf{x})$")

contours = ax.contour(X1, X2, r.logit(X_grid).numpy(), cmap="viridis")

fig.colorbar(contours, ax=ax)
ax.clabel(contours, fmt="%.2f")

ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')

plt.show()

# %%

fig, ax = plt.subplots()

ax.set_title(r"$r(\mathbf{x}) = \exp(f(\mathbf{x}))$")

contours = ax.contour(X1, X2, r.ratio(X_grid).numpy(), cmap="viridis")

fig.colorbar(contours, ax=ax)
ax.clabel(contours, fmt="%.2f")

ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')

plt.show()

# %%

fig, ax = plt.subplots()

ax.set_title(r"$P(s=1|\mathbf{x}) = \sigma(f(\mathbf{x}))$")

contours = ax.contour(X1, X2, r.prob(X_grid).numpy(), cmap="viridis")

ax.scatter(*X.T, c=r.prob(X).numpy(), cmap="viridis", alpha=0.6)

fig.colorbar(contours, ax=ax)
ax.clabel(contours, fmt="%.2f")

ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')

plt.show()
# %%

fig, ax = plt.subplots()

ax.scatter(*X_train.T, c=y_train, cmap="RdYlBu", alpha=0.8, label="train")
ax.scatter(*X_test.T, marker='x', c=y_test, cmap="RdYlBu", alpha=0.2, label="test")

contours = ax.contour(X1, X2, class_posterior(X1, X2).numpy(), cmap="RdYlBu")

fig.colorbar(contours, ax=ax)
ax.clabel(contours, fmt="%.2f")

ax.legend()

ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')

plt.show()

# %%
# Linear Logistic Regression
# --------------------------

model = LogisticRegression(C=C, random_state=seed)
model.fit(X_train, y_train)
model.score(X_test, y_test)

# %%

p_grid = model.predict_proba(X_grid.reshape(-1, 2)) \
              .reshape(200, 200, 2)

# %%

fig, ax = plt.subplots()

ax.set_title("Without importance sampling (uniform weights)")

contours = ax.contour(X1, X2, p_grid[..., -1], cmap="RdYlBu")

fig.colorbar(contours, ax=ax)
ax.clabel(contours, fmt="%.2f")

ax.scatter(*X_train.T, c=y_train, cmap="RdYlBu", alpha=0.8, label="train")
ax.scatter(*X_test.T, marker='x', c=y_test, cmap="RdYlBu", alpha=0.2, label="test")

ax.legend()

ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')

plt.show()

# %%

model = LogisticRegression(C=C, random_state=seed)
model.fit(X_train, y_train, sample_weight=r.ratio(X_train).numpy())
model.score(X_test, y_test)

# %%

p_grid = model.predict_proba(X_grid.reshape(-1, 2)) \
              .reshape(200, 200, 2)

# %%

fig, ax = plt.subplots()

ax.set_title("With importance sampling (exact density ratio)")

contours = ax.contour(X1, X2, p_grid[..., -1], cmap="RdYlBu")

fig.colorbar(contours, ax=ax)
ax.clabel(contours, fmt="%.2f")

ax.scatter(*X_train.T, c=y_train, s=r.ratio(X_train).numpy(),
           cmap="RdYlBu", alpha=0.8, label="train")
ax.scatter(*X_test.T, marker='x', c=y_test, s=r.ratio(X_test).numpy(),
           cmap="RdYlBu", alpha=0.2, label="test")

ax.legend()

ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')

plt.show()
