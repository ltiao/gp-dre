# -*- coding: utf-8 -*-
"""
Synthetic 2D Classification Covariate Shift Problem
===================================================
"""
# sphinx_gallery_thumbnail_number = 8

import numpy as np
import tensorflow_probability as tfp

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from gpdre import DensityRatioMarginals
from gpdre.models import DenseSequential

from sklearn.svm import SVC
from sklearn.datasets import make_moons

from tensorflow.keras.regularizers import l2

# %%

# shortcuts
tfd = tfp.distributions
kernels = tfp.math.psd_kernels

# constants
num_samples = 1000

num_layers = 2
num_units = 32
activation = "relu"
l2_factor = 5e-3

optimizer = "adam"
epochs = 400
batch_size = 32

seed = 8888
dataset_seed = 42

X2, X1 = np.mgrid[-1.0:1.5:200j, -1.5:2.5:200j]
X_grid = np.dstack((X1, X2))

# %%

X, y = make_moons(num_samples, noise=0.05, random_state=dataset_seed)

# %%

fig, ax = plt.subplots()

ax.scatter(*X.T, c=y, cmap="RdYlBu", alpha=0.6)

ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')

plt.show()

# %%

test = tfd.MultivariateNormalDiag(loc=[0.5, 0.25], scale_diag=[0.5, 0.5])
train = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(probs=[0.5, 0.5]),
    components_distribution=tfd.MultivariateNormalDiag(
        loc=[[-1., -0.5], [2., 1.0]], scale_diag=[0.5, 1.5])
)
r = DensityRatioMarginals(top=test, bot=train)

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

(X_train, y_train), (X_val, y_val) = r.train_test_split(X, y, seed=seed)

# %%

fig, ax = plt.subplots()

ax.scatter(*X_train.T, c=y_train, cmap="RdYlBu", alpha=0.8, label="train")
ax.scatter(*X_val.T, marker='x', c=y_val, cmap="RdYlBu", alpha=0.2, label="test")

ax.legend()

ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')

plt.show()

# %%
# Support Vector Machine Classifier
# ---------------------------------

model = SVC(C=1.0, kernel="rbf", gamma="scale", max_iter=-1, probability=True,
            random_state=seed)
model.fit(X_train, y_train)
model.score(X_val, y_val)

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
ax.scatter(*X_val.T, marker='x', c=y_val, cmap="RdYlBu", alpha=0.2, label="test")

ax.legend()

ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')

plt.show()

# %%

model = SVC(C=1.0, kernel="rbf", gamma="scale", max_iter=-1, probability=True,
            random_state=seed)
model.fit(X_train, y_train, sample_weight=r.ratio(X_train).numpy())
model.score(X_val, y_val)

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
ax.scatter(*X_val.T, marker='x', c=y_val, s=r.ratio(X_val).numpy(),
           cmap="RdYlBu", alpha=0.2, label="test")

ax.legend()

ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')

plt.show()

# %%
# Multi-layer Perceptron Classifier
# ---------------------------------

model = DenseSequential(1, num_layers=num_layers, num_units=num_units,
                        layer_kws=dict(activation=activation,
                                       kernel_regularizer=l2(l2_factor)),
                        final_layer_kws=dict(activation="sigmoid"))
model.compile(optimizer=optimizer, loss="binary_crossentropy",
              metrics=["accuracy"])
hist_uniform = model.fit(X_train, y_train, epochs=epochs,
                         batch_size=batch_size,
                         validation_data=(X_val, y_val))
# %%

val_loss, val_accuracy = model.evaluate(X_val, y_val)
val_accuracy

# %%

p_grid = model.predict(X_grid)

# %%

fig, ax = plt.subplots()

ax.set_title("Without importance sampling (uniform weights)")

contours = ax.contour(X1, X2, p_grid.squeeze(), cmap="RdYlBu")

fig.colorbar(contours, ax=ax)
ax.clabel(contours, fmt="%.2f")

ax.scatter(*X_train.T, c=y_train, cmap="RdYlBu", alpha=0.8, label="train")
ax.scatter(*X_val.T, marker='x', c=y_val, cmap="RdYlBu", alpha=0.2, label="test")

ax.legend()

ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')

plt.show()

# %%

model = DenseSequential(1, num_layers=num_layers, num_units=num_units,
                        layer_kws=dict(activation=activation,
                                       kernel_regularizer=l2(l2_factor)),
                        final_layer_kws=dict(activation="sigmoid"))
model.compile(optimizer=optimizer, loss="binary_crossentropy",
              metrics=["accuracy"])
hist_exact = model.fit(X_train, y_train, epochs=epochs,
                       batch_size=batch_size,
                       validation_data=(X_val, y_val),
                       sample_weight=r.ratio(X_train).numpy())
# %%

val_loss, val_accuracy = model.evaluate(X_val, y_val)
val_accuracy

# %%

p_grid = model.predict(X_grid)

# %%

fig, ax = plt.subplots()

ax.set_title("With importance sampling (exact density ratio)")

contours = ax.contour(X1, X2, p_grid.squeeze(), cmap="RdYlBu")

fig.colorbar(contours, ax=ax)
ax.clabel(contours, fmt="%.2f")

ax.scatter(*X_train.T, c=y_train, s=r.ratio(X_train).numpy(),
           cmap="RdYlBu", alpha=0.8, label="train")
ax.scatter(*X_val.T, marker='x', c=y_val, s=r.ratio(X_val).numpy(),
           cmap="RdYlBu", alpha=0.2, label="test")

ax.legend()

ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')

plt.show()

# %%

rows = []
rows.append(pd.DataFrame(hist_exact.history).assign(weight="exact"))
rows.append(pd.DataFrame(hist_uniform.history).assign(weight="uniform"))

data = pd.concat(rows, axis="index", sort=True)
data.index.name = "epoch"
data.reset_index(inplace=True)
data.rename(lambda s: s.replace('_', ' '), axis="columns", inplace=True)

# %%

fig, ax = plt.subplots()

sns.lineplot(x="epoch", y="val accuracy", hue="weight", data=data, ax=ax)

plt.show()
