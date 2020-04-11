# -*- coding: utf-8 -*-
"""
Synthetic 1D Toy Problem
========================
"""
# sphinx_gallery_thumbnail_number = 8

import numpy as np

import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from gpdre import GaussianProcessDensityRatioEstimator
from gpdre.datasets import make_classification_dataset
from gpdre.initializers import KMeans
from gpdre.plotting import fill_between_stddev
from gpdre.utils import DensityRatio

# %%

# shortcuts
tfd = tfp.distributions
kernels = tfp.math.psd_kernels


# constants
num_train = 2000  # nbr training points in synthetic dataset
num_inducing_points = 50
num_features = 1  # dimensionality

num_test = 40
num_index_points = 256  # nbr of index points
num_samples = 25

optimizer = "adam"
num_epochs = 800
batch_size = 64
shuffle_buffer_size = 500
quadrature_size = 20

jitter = 1e-6

kernel_cls = kernels.MaternFiveHalves

seed = 8888  # set random seed for reproducibility
random_state = np.random.RandomState(seed)

x_min, x_max = -5.0, 5.0
y_min, y_max = 0.0, 12.0

# index points
X_grid = np.linspace(x_min, x_max, num_index_points).reshape(-1, num_features)

golden_ratio = 0.5 * (1 + np.sqrt(5))

# %%
# Synthetic dataset
# -----------------

p = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(probs=[0.3, 0.7]),
    components_distribution=tfd.Normal(loc=[2.0, -3.0], scale=[1.0, 0.5]))
q = tfd.Normal(loc=0.0, scale=2.0)
r = DensityRatio(top=p, bot=q)

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

ax.plot(X_grid, r(X_grid), c='k', label=r"$r(x) = \exp{f(x)}$")

ax.set_xlabel('$x$')
ax.set_ylabel('$r(x)$')

ax.legend()

plt.show()

# %%
# Create classification dataset.

X_p, X_q = r.make_dataset(num_train, seed=seed)
X_train, y_train = make_classification_dataset(X_p, X_q)

# %%
# Dataset visualized against the Bayes optimal classifier.

fig, ax = plt.subplots()

ax.plot(X_grid, r.optimal_score(X_grid), c='k',
        label=r"$\pi(x) = \sigma(f(x))$")
ax.scatter(X_train, y_train, c=y_train, s=12.**2,
           marker='s', alpha=0.1, cmap="coolwarm_r")
ax.set_yticks([0, 1])
ax.set_yticklabels([r"$x_q \sim q(x)$", r"$x_p \sim p(x)$"])
ax.set_xlabel('$x$')

ax.legend()

plt.show()

# %%

gpdre = GaussianProcessDensityRatioEstimator(
    input_dim=num_features,
    num_inducing_points=num_inducing_points,
    inducing_index_points_initializer=KMeans(X_train, seed=seed),
    kernel_cls=kernel_cls, jitter=jitter, seed=seed)
gpdre.compile(optimizer=optimizer, quadrature_size=quadrature_size)
gpdre.fit(X_p, X_q, epochs=num_epochs, batch_size=batch_size,
          buffer_size=shuffle_buffer_size)

# %%

log_ratio_estimator = gpdre.logit(X_grid)

# %%

fig, ax = plt.subplots()

ax.plot(X_grid, r.logit(X_grid), c='k',
        label=r"$f(x) = \log p(x) - \log q(x)$")

ax.plot(X_grid, log_ratio_estimator.mean().numpy().T,
        label="posterior mean")
fill_between_stddev(X_grid.squeeze(),
                    log_ratio_estimator.mean().numpy().squeeze(),
                    log_ratio_estimator.stddev().numpy().squeeze(), alpha=0.1,
                    label="posterior std dev", ax=ax)

ax.scatter(gpdre.inducing_index_points.numpy(),
           np.full_like(gpdre.inducing_index_points.numpy(), -5.0),
           marker='^', c="tab:gray", label="inducing inputs", alpha=0.8)
ax.scatter(gpdre.inducing_index_points.numpy(),
           gpdre.variational_inducing_observations_loc.numpy(),
           marker='+', c="tab:blue", label="inducing variable mean")

ax.set_xlabel('$x$')
ax.set_ylabel('$f(x)$')

ax.legend()

plt.show()

# %%

ratio_estimator = gpdre(X_grid)

# %%

fig, ax = plt.subplots()

ax.plot(X_grid, r(X_grid), c='k', label=r"$r(x) = \exp{f(x)}$")

ax.plot(X_grid, ratio_estimator.distribution.quantile(0.5),
        c="tab:blue", label="median")

ax.fill_between(X_grid.squeeze(),
                ratio_estimator.distribution.quantile(0.25),
                ratio_estimator.distribution.quantile(0.75),
                alpha=0.1, label="quartiles")

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$r(x)$")

ax.legend()

plt.show()

# %%

ratio_estimator_marginal = gpdre(X_grid, reinterpreted_batch_ndims=None)

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

ax.plot(X_grid, gpdre.predictive_sample(X_grid, num_samples).mean().numpy().T,
        color="tab:blue", linewidth=0.4, alpha=0.6)
ax.plot(X_grid, r.optimal_score(X_grid), c='k',
        label=r"$\pi(x) = \sigma(f(x))$")
ax.scatter(X_train, y_train, c=y_train, s=12.**2,
           marker='s', alpha=0.1, cmap="coolwarm_r")
ax.set_yticks([0, 1])
ax.set_yticklabels([r"$x_q \sim q(x)$", r"$x_p \sim p(x)$"])
ax.set_xlabel('$x$')

ax.legend()

plt.show()

# %%

X_test, y_test = r.make_classification_dataset(num_test, seed=seed)

# %%

y_scores = gpdre.predictive_sample(X_test, num_samples).mean().numpy()
y_score_min, y_score_max = np.percentile(y_scores, q=[5.0, 95.0], axis=0)
# %%

fig, ax = plt.subplots()

ax.plot(X_grid, r.optimal_score(X_grid), c='k', label=r"$\pi(x) = \sigma(f(x))$")

ax.scatter(X_test, np.median(y_scores, axis=0), c="tab:blue", label="median",
           alpha=0.8)
ax.vlines(X_test, ymin=y_score_min, ymax=y_score_max, color="tab:blue",
          label=r"90\% confidence interval", alpha=0.8)

ax.scatter(X_test, y_test, c=y_test, s=12.**2,
           marker='s', alpha=0.1, cmap="coolwarm_r")

ax.set_ylabel(r"$\pi(x)$")
ax.set_xlabel(r"$x$")
ax.set_xlim(x_min, x_max)

ax.legend()

plt.show()