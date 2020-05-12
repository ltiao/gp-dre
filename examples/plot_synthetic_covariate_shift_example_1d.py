# -*- coding: utf-8 -*-
"""
Synthetic 1D Regression Covariate Shift Problem
===============================================
"""
# sphinx_gallery_thumbnail_number = 6

import numpy as np
import tensorflow_probability as tfp

import matplotlib.pyplot as plt

from gpdre import DensityRatioMarginals
from gpdre.datasets import make_classification_dataset
from gpdre.metrics import normalized_mean_squared_error

from sklearn.linear_model import LinearRegression
# %%

# shortcuts
tfd = tfp.distributions

# constants
num_features = 1  # dimensionality
num_train = 100  # nbr training points in synthetic dataset
num_test = 100
num_index_points = 512  # nbr of index points

dataset_seed = 42  # set random seed for reproducibility

x_min, x_max = -1.5, 2.0
y_min, y_max = 0.0, 5.0

# index points
X_grid = np.linspace(x_min, x_max, num_index_points).reshape(-1, num_features)
y_num_points = 512


# %%
def poly(x):

    return - x + x**3


# %%
test = tfd.Normal(loc=0.0, scale=0.3)
train = tfd.Normal(loc=0.5, scale=0.5)

r = DensityRatioMarginals(top=test, bot=train)

# %%
(X_train, y_train), (X_test, y_test) = r.make_regression_dataset(
    num_test, num_train, latent_fn=poly,
    noise_scale=0.3, seed=dataset_seed)

# %%
X, s = make_classification_dataset(X_test, X_train)
S = np.atleast_2d(s).T

# %%
fig, ax = plt.subplots()

ax.plot(X_grid, poly(X_grid), label="true", color="tab:gray")
ax.scatter(X_train, y_train, alpha=0.6, label="train")
ax.scatter(X_test, y_test, marker='x', alpha=0.6, label="test")

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

plt.show()

# %%
fig, ax = plt.subplots()

ax.plot(X_grid, r.bot.prob(X_grid), label=r"$p_{\mathrm{tr}}(\mathbf{x})$")
ax.plot(X_grid, r.top.prob(X_grid), label=r"$p_{\mathrm{te}}(\mathbf{x})$")

ax.set_xlabel('$x$')
ax.set_ylabel('density')

ax.legend()

plt.show()

# %%
fig, ax = plt.subplots()

ax.plot(X_grid, r.logit(X_grid), c='k',
        label=r"$f(x) = \log p(x) - \log q(x)$")

ax.set_xlabel('$x$')
ax.set_ylabel('$f(x)$')

ax.legend()

plt.show()

# %%
fig, ax = plt.subplots()

ax.plot(X_grid, r.ratio(X_grid), c='k', label=r"$r(x) = \exp{f(x)}$")

ax.set_xlabel('$x$')
ax.set_ylabel('$r(x)$')

ax.legend()

plt.show()

# %%
fig, ax = plt.subplots()

ax.plot(X_grid, r.prob(X_grid), c='k',
        label=r"$\pi(x) = \sigma(f(x))$")
ax.scatter(X, S, c=S, s=12.**2, marker='s', alpha=0.1, cmap="coolwarm_r")
ax.set_yticks([0, 1])
ax.set_yticklabels([r"$x_q \sim q(x)$", r"$x_p \sim p(x)$"])
ax.set_xlabel('$x$')

ax.legend()

plt.show()

# %%
# Ordinary Linear Regression
# --------------------------

model_uniform = LinearRegression()
model_uniform.fit(X_train, y_train)
y_pred = model_uniform.predict(X_test)

nmse_uniform = normalized_mean_squared_error(y_test, y_pred)
nmse_uniform
# %%

model_exact = LinearRegression()
model_exact.fit(X_train, y_train,
                sample_weight=r.ratio(X_train).numpy().squeeze())
y_pred = model_exact.predict(X_test)

nmse_exact = normalized_mean_squared_error(y_test, y_pred)
nmse_exact
# %%

fig, ax = plt.subplots(figsize=(6, 6))

ax.plot(X_grid, poly(X_grid), label="true", color="tab:gray")
ax.plot(X_grid, model_uniform.predict(X_grid), label="uniform")
ax.plot(X_grid, model_exact.predict(X_grid), label="exact")

ax.scatter(X_train, y_train, alpha=0.6, label="train")
ax.scatter(X_test, y_test, marker='x', alpha=0.6, label="test")

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')

ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-1.0, 1.5)

ax.legend()

plt.show()
# %%

import tensorflow as tf
import gpflow

from gpdre.plotting import fill_between_stddev
# %%

jitter = 1e-6
kernel_cls = gpflow.kernels.SquaredExponential
# %%

vgp = gpflow.models.VGP(
    data=(X, S),
    likelihood=gpflow.likelihoods.Bernoulli(invlink=tf.sigmoid),
    kernel=kernel_cls()
)
optimizer = gpflow.optimizers.Scipy()
optimizer.minimize(vgp.training_loss, variables=vgp.trainable_variables)
# %%

qf_loc, qf_var = vgp.predict_f(X_grid, full_cov=False)
qf_scale = tf.sqrt(qf_var)
# %%

K = vgp.kernel.K(X) + jitter * tf.eye(num_train + num_test, dtype=tf.float64)
L = tf.linalg.cholesky(K)
m = tf.matmul(L, vgp.q_mu)
# %%

fig, ax = plt.subplots()

ax.plot(X_grid, r.logit(X_grid), c='k',
        label=r"$f(x) = \log p(x) - \log q(x)$")

ax.plot(X_grid, qf_loc.numpy(), label="posterior mean")
fill_between_stddev(X_grid.squeeze(),
                    qf_loc.numpy().squeeze(),
                    qf_scale.numpy().squeeze(), alpha=0.1,
                    label="posterior std dev", ax=ax)

ax.scatter(X, m, marker='+', c="tab:blue",
           label="inducing variable mean")

ax.set_xlabel('$x$')
ax.set_ylabel('$f(x)$')

ax.legend()

plt.show()
# %%

ratio_marginal = tfd.LogNormal(loc=tf.squeeze(qf_loc, axis=-1),
                               scale=tf.squeeze(qf_scale, axis=-1))
ratio = tfd.Independent(ratio_marginal, reinterpreted_batch_ndims=1)
# %%

fig, ax = plt.subplots()

ax.plot(X_grid, r(X_grid), c='k', label=r"$r(x) = \exp{f(x)}$")

ax.plot(X_grid, ratio_marginal.quantile(0.5),
        c="tab:blue", label="median")

ax.fill_between(X_grid.squeeze(),
                ratio_marginal.quantile(0.25),
                ratio_marginal.quantile(0.75),
                alpha=0.1, label="interquartile range")

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$r(x)$")

ax.legend()

plt.show()
# %%

Y_grid = np.linspace(y_min, y_max, y_num_points).reshape(-1, 1)
x, y = np.meshgrid(X_grid, Y_grid)
# %%

fig, ax = plt.subplots()

contours = ax.pcolormesh(x, y, ratio_marginal.prob(Y_grid),
                         vmax=1.0, cmap="Blues")

ax.plot(X_grid, r(X_grid), c='k', label=r"$r(x) = \exp{f(x)}$")

fig.colorbar(contours, extend="max", ax=ax)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$r(x)$")

ax.legend()

plt.show()
# %%

fig, ax = plt.subplots()

contours = ax.pcolormesh(x, y, ratio_marginal.prob(Y_grid),
                         vmax=1.0, cmap="Blues")

ax.plot(X_grid, r(X_grid), c='k', label=r"$r(x) = \exp{f(x)}$")

ax.plot(X_grid, ratio_marginal.mode(), c="tab:blue",
        label="transformed posterior mode")
ax.plot(X_grid, ratio_marginal.mean(), c="tab:blue", linestyle="--",
        label="transformed posterior mean")

fig.colorbar(contours, extend="max", ax=ax)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$r(x)$")

ax.set_ylim(y_min, y_max)

ax.legend()

plt.show()
