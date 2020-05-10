# -*- coding: utf-8 -*-
"""
Sparse VGP (GPFlow)
===================

"""
# sphinx_gallery_thumbnail_number = 8

import numpy as np

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib.pyplot as plt

from tensorflow.keras import optimizers

from gpdre.datasets import make_classification_dataset
from gpdre.initializers import KMeans
from gpdre.plotting import fill_between_stddev
from gpdre.base import DensityRatioMarginals

from tqdm import trange

# %%

# shortcuts
tfd = tfp.distributions

# constants
num_train = 2000  # nbr training points in synthetic dataset
num_inducing_points = 50
num_features = 1  # dimensionality

num_test = 40
num_index_points = 512  # nbr of index points
num_samples = 25

optimizer_name = "adam"
num_epochs = 800
batch_size = 64
shuffle_buffer_size = 500

whiten = True

kernel_cls = gpflow.kernels.SquaredExponential

jitter = 1e-6
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

inducing_index_points_initializer = KMeans(X_train, seed=seed)
inducing_index_points_initial = (
    inducing_index_points_initializer(shape=(num_inducing_points, num_features)))

# %%

vgp = gpflow.models.SVGP(
    likelihood=gpflow.likelihoods.Bernoulli(invlink=tf.sigmoid),
    inducing_variable=inducing_index_points_initial,
    kernel=kernel_cls(), num_data=len(X_train), whiten=whiten,
)

# %%

optimizer = optimizers.get(optimizer_name)

# %%
@tf.function
def train_on_batch(X_batch, y_batch):

    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(vgp.trainable_variables)
        loss = vgp.training_loss((X_batch, y_batch))

    gradients = tape.gradient(loss, vgp.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vgp.trainable_variables))
# %%


dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)) \
                         .shuffle(seed=seed, buffer_size=shuffle_buffer_size) \
                         .batch(batch_size, drop_remainder=True)
# %%

for epoch in trange(num_epochs):
    for step, (X_batch, y_batch) in enumerate(dataset):

        train_on_batch(X_batch, y_batch)
# %%
qf_loc, qf_var = vgp.predict_f(X_grid, full_cov=False)
qf_scale = tf.sqrt(qf_var)
# %%

K = (vgp.kernel.K(vgp.inducing_variable.Z) +
     jitter * tf.eye(num_inducing_points, dtype=tf.float64))
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

ax.scatter(vgp.inducing_variable.Z.numpy(),
           np.full_like(vgp.inducing_variable.Z.numpy(), -5.0),
           marker='^', c="tab:gray", label="inducing inputs", alpha=0.8)
ax.scatter(vgp.inducing_variable.Z.numpy(), m.numpy(),
           marker='+', c="tab:blue", label="inducing variable mean")

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

# %%

qf_samples = vgp.predict_f_samples(X_grid, num_samples)
py = tfd.Independent(
    tfd.Bernoulli(logits=tf.squeeze(qf_samples, axis=-1)),
    reinterpreted_batch_ndims=1)

# %%

fig, ax = plt.subplots()

ax.plot(X_grid, py.mean().numpy().T,
        color="tab:blue", linewidth=0.4, alpha=0.6)
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

X_test, y_test = r.make_classification_dataset(num_test, seed=seed)

# %%

qf_samples = vgp.predict_f_samples(X_test, num_samples)
py = tfd.Independent(
    tfd.Bernoulli(logits=tf.squeeze(qf_samples, axis=-1)),
    reinterpreted_batch_ndims=1)
y_scores = py.mean().numpy()
y_score_min, y_score_max = np.percentile(y_scores, q=[5.0, 95.0], axis=0)
# %%

fig, ax = plt.subplots()

ax.plot(X_grid, r.prob(X_grid), c='k', label=r"$\pi(x) = \sigma(f(x))$")

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
