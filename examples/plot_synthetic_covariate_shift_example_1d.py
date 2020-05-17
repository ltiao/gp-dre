# -*- coding: utf-8 -*-
"""
Synthetic 1D Regression Covariate Shift Problem
===============================================
"""
# sphinx_gallery_thumbnail_number = 6

import numpy as np
import tensorflow_probability as tfp

import matplotlib.pyplot as plt

from gpdre import DensityRatioMarginals, GaussianProcessDensityRatioEstimator
from gpdre.datasets import make_classification_dataset
from gpdre.metrics import normalized_mean_squared_error
from gpdre.plotting import fill_between_stddev

from gpflow.models import VGP
from gpflow.kernels import Matern52
from gpflow.optimizers import Scipy

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
# %%

# shortcuts
tfd = tfp.distributions

# constants
num_features = 1  # dimensionality
num_train = 100  # nbr training points in synthetic dataset
num_test = 100
num_index_points = 512  # nbr of index points
num_samples = 25
num_inducing_points = 50
jitter = 1e-6

kernel_cls = Matern52
optimizer = Scipy()

seed = 8888
dataset_seed = 24  # set random seed for reproducibility

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
ax.scatter(X, s, c=s, s=12.**2, marker='s', alpha=0.1, cmap="coolwarm_r")
ax.set_yticks([0, 1])
ax.set_yticklabels([r"$x_q \sim q(x)$", r"$x_p \sim p(x)$"])
ax.set_xlabel('$x$')

ax.legend()

plt.show()

# %%
# Ordinary Least Squares (OLS)
# ----------------------------

model_train_uniform = LinearRegression()
model_train_uniform.fit(X_train, y_train)
y_pred = model_train_uniform.predict(X_test)

nmse_train_uniform = normalized_mean_squared_error(y_test, y_pred)
nmse_train_uniform
# %%

model_test_uniform = LinearRegression()
model_test_uniform.fit(X_test, y_test)
y_pred = model_test_uniform.predict(X_test)

nmse_test_uniform = normalized_mean_squared_error(y_test, y_pred)
nmse_test_uniform
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
ax.plot(X_grid, model_train_uniform.predict(X_grid), label="uniform train")
ax.plot(X_grid, model_test_uniform.predict(X_grid), label="uniform test")
ax.plot(X_grid, model_exact.predict(X_grid), label="exact train")

ax.scatter(X_train, y_train, alpha=0.6, label="train")
ax.scatter(X_test, y_test, marker='x', alpha=0.6, label="test")

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')

ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-1.0, 1.5)

ax.legend()

plt.show()
# %%
# Gaussian Process Density Ratio Estimation
# -----------------------------------------

gpdre = GaussianProcessDensityRatioEstimator(kernel_cls=kernel_cls,
                                             vgp_cls=VGP,
                                             jitter=jitter,
                                             seed=seed)
gpdre.compile(optimizer=optimizer)
gpdre.fit(X_test, X_train)
# %%

log_ratio_mean = gpdre.logit(X_grid, convert_to_tensor_fn=tfd.Distribution.mean)
log_ratio_stddev = gpdre.logit(X_grid, convert_to_tensor_fn=tfd.Distribution.stddev)

# %%

fig, ax = plt.subplots()

ax.plot(X_grid, r.logit(X_grid), c='k',
        label=r"$f(x) = \log p(x) - \log q(x)$")

ax.plot(X_grid, log_ratio_mean.numpy().T,
        label="posterior mean")
fill_between_stddev(X_grid.squeeze(),
                    log_ratio_mean.numpy().squeeze(),
                    log_ratio_stddev.numpy().squeeze(), alpha=0.1,
                    label="posterior std dev", ax=ax)

ax.set_xlabel('$x$')
ax.set_ylabel('$f(x)$')

ax.legend()

plt.show()
# %%

ratio_estimator = gpdre.ratio_distribution(X_grid)

# %%

fig, ax = plt.subplots()

ax.plot(X_grid, r(X_grid), c='k', label=r"$r(x) = \exp{f(x)}$")

ax.plot(X_grid, ratio_estimator.distribution.quantile(0.5),
        c="tab:blue", label="median")

ax.fill_between(X_grid.squeeze(),
                ratio_estimator.distribution.quantile(0.25),
                ratio_estimator.distribution.quantile(0.75),
                alpha=0.1, label="interquartile range")

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$r(x)$")

ax.legend()

plt.show()

# %%

ratio_estimator_marginal = gpdre.ratio_distribution(X_grid,
                                                    reinterpreted_batch_ndims=None)

y_num_points = 512
Y_grid = np.linspace(y_min, y_max, y_num_points).reshape(-1, 1)
x, y = np.meshgrid(X_grid, Y_grid)

# %%

fig, ax = plt.subplots()

contours = ax.pcolormesh(x, y, ratio_estimator_marginal.prob(Y_grid),
                         vmax=1.0, cmap="Blues")

ax.plot(X_grid, r(X_grid), c='k', label=r"$r(x) = \exp{f(x)}$")

fig.colorbar(contours, extend="max", ax=ax)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$r(x)$")

ax.legend()

plt.show()

# %%

fig, ax = plt.subplots()

contours = ax.pcolormesh(x, y, ratio_estimator_marginal.prob(Y_grid),
                         vmax=1.0, cmap="Blues")

ax.plot(X_grid, r(X_grid), c='k', label=r"$r(x) = \exp{f(x)}$")

ax.plot(X_grid, ratio_estimator_marginal.mode(), c="tab:blue",
        label="transformed posterior mode")
ax.plot(X_grid, ratio_estimator_marginal.mean(), c="tab:blue", linestyle="--",
        label="transformed posterior mean")

fig.colorbar(contours, extend="max", ax=ax)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$r(x)$")

ax.legend()

plt.show()

# %%

fig, ax = plt.subplots()

ax.plot(X_grid, gpdre.conditional(X_grid, num_samples).mean().numpy().T,
        color="tab:blue", linewidth=0.4, alpha=0.6)
ax.plot(X_grid, r.prob(X_grid), c='k',
        label=r"$\pi(x) = \sigma(f(x))$")
ax.scatter(X, s, c=s, s=12.**2, marker='s', alpha=0.1, cmap="coolwarm_r")
ax.set_yticks([0, 1])
ax.set_yticklabels([r"$x_q \sim q(x)$", r"$x_p \sim p(x)$"])
ax.set_xlabel('$x$')

ax.legend()

plt.show()
# %%


def regression_metric(X_train, y_train, X_test, y_test, sample_weight=None):

    feature_scaler = StandardScaler()

    Z_train = feature_scaler.fit_transform(X_train)
    Z_test = feature_scaler.transform(X_test)

    model = LinearRegression()
    model.fit(Z_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(Z_test)

    return normalized_mean_squared_error(y_test, y_pred)


# %%
import tensorflow as tf
import pandas as pd
import seaborn as sns

from gpflow.kernels import SquaredExponential, Matern12, Matern32, Matern52
# %%

kernels = {
    "squared_exponental": SquaredExponential,
    "matern12": Matern12,
    "matern32": Matern32,
    "matern52": Matern52
}

num_seeds = 5
rows = []

for seed in range(num_seeds):

    # # Uniform
    # nmse = regression_metric(X_train, y_train, X_test, y_test)
    # rows.append(dict(weight="uniform", seed=seed, nmse=nmse))

    # # Exact
    # nmse = regression_metric(X_train, y_train, X_test, y_test,
    #                          sample_weight=r.ratio(X_train).numpy().squeeze())
    # rows.append(dict(weight="exact", seed=seed, nmse=nmse))

    for kernel_name, kernel_cls in kernels.items():

        # GP
        gpdre = GaussianProcessDensityRatioEstimator(kernel_cls=kernel_cls,
                                                     vgp_cls=VGP,
                                                     jitter=jitter,
                                                     seed=seed)
        gpdre.compile(optimizer=optimizer)
        gpdre.fit(X_test, X_train)

        r_mean = gpdre.ratio(X_train, convert_to_tensor_fn=tfd.Distribution.mean)
        nmse = regression_metric(X_train, y_train, X_test, y_test,
                                 sample_weight=r_mean.numpy().squeeze())
        rows.append(dict(kernel_name=kernel_name, weight="mean",
                         seed=seed, nmse=nmse))

        r_mode = gpdre.ratio(X_train, convert_to_tensor_fn=tfd.Distribution.mode)
        nmse = regression_metric(X_train, y_train, X_test, y_test,
                                 sample_weight=r_mode.numpy().squeeze())
        rows.append(dict(kernel_name=kernel_name, weight="mode",
                         seed=seed, nmse=nmse))

        r_median = gpdre.ratio(X_train, convert_to_tensor_fn=lambda d: d.distribution.quantile(0.5))
        nmse = regression_metric(X_train, y_train, X_test, y_test,
                                 sample_weight=r_median.numpy().squeeze())
        rows.append(dict(kernel_name=kernel_name, weight="median",
                         seed=seed, nmse=nmse))

        r_sample = tf.exp(gpdre.logit(X_train, convert_to_tensor_fn=tfd.Distribution.sample))
        nmse = regression_metric(X_train, y_train, X_test, y_test,
                                 sample_weight=r_sample.numpy().squeeze())
        rows.append(dict(kernel_name=kernel_name, weight="sample",
                         seed=seed, nmse=nmse))
# %%

data = pd.DataFrame(rows)
data
# %%

fig, ax = plt.subplots()

sns.stripplot(x="kernel_name", y="nmse", hue="weight", alpha=0.6,
              jitter=False, dodge=True, data=data, ax=ax)

# sns.pointplot(x="split", y="nmse", data=data_diff.reset_index(),
#               palette="tab20", join=False, ci=None, markers='d',
#               scale=0.75, ax=ax)
ax.set_yscale("log")

plt.show()
# %%

fig, ax = plt.subplots()

sns.stripplot(x="weight", y="nmse", hue="kernel_name", alpha=0.6,
              jitter=False, dodge=True, data=data, ax=ax)

# sns.pointplot(x="split", y="nmse", data=data_diff.reset_index(),
#               palette="tab20", join=False, ci=None, markers='d',
#               scale=0.75, ax=ax)
ax.set_yscale("log")

plt.show()
