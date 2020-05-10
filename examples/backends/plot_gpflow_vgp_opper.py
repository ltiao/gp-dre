# -*- coding: utf-8 -*-
"""
VGP -- Opper and Archambeau 2009 (GPFlow)
=========================================

"""
# sphinx_gallery_thumbnail_number = 8

import numpy as np

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib.pyplot as plt

from gpdre.datasets import make_classification_dataset
from gpdre.plotting import fill_between_stddev
from gpdre.base import DensityRatioMarginals

# %%

# shortcuts
tfd = tfp.distributions

# constants
num_train = 256  # nbr training points in synthetic dataset
num_features = 1  # dimensionality

num_test = 40
num_index_points = 512  # nbr of index points
num_samples = 25

kernel_cls = gpflow.kernels.SquaredExponential

seed = 8888  # set random seed for reproducibility
random_state = np.random.RandomState(seed)

x_min, x_max = -5.0, 5.0
y_min, y_max = 0.0, 12.0

# index points
X_grid = np.linspace(x_min, x_max, num_index_points).reshape(-1, num_features)
y_num_points = 512

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

X_p, X_q = r.make_dataset(num_train, seed=seed)
X_train, y_train = make_classification_dataset(X_p, X_q)
Y_train = np.atleast_2d(y_train).T

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
vgp = gpflow.models.VGPOpperArchambeau(
    data=(X_train, Y_train),
    likelihood=gpflow.likelihoods.Bernoulli(invlink=tf.sigmoid),
    kernel=kernel_cls()
)

# %%

optimizer = gpflow.optimizers.Scipy()
optimizer.minimize(vgp.training_loss, variables=vgp.trainable_variables)

# %%
qf_loc, qf_var = vgp.predict_f(X_grid, full_cov=False)
qf_scale = tf.sqrt(qf_var)
# %%

m = tf.matmul(vgp.kernel.K(X_train), vgp.q_alpha)
# %%

fig, ax = plt.subplots()

ax.plot(X_grid, r.logit(X_grid), c='k',
        label=r"$f(x) = \log p(x) - \log q(x)$")

ax.plot(X_grid, qf_loc.numpy(), label="posterior mean")
fill_between_stddev(X_grid.squeeze(),
                    qf_loc.numpy().squeeze(),
                    qf_scale.numpy().squeeze(), alpha=0.1,
                    label="posterior std dev", ax=ax)

ax.scatter(X_train, m.numpy(), marker='+', c="tab:blue",
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

ax.legend()

plt.show()
