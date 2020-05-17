# -*- coding: utf-8 -*-
"""
Synthetic 2D Classification Covariate Shift Problem
===================================================
"""
# sphinx_gallery_thumbnail_number = 8

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.backend as K

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from gpdre import DensityRatioMarginals, GaussianProcessDensityRatioEstimator
from gpdre.base import MLPDensityRatioEstimator
from gpdre.external.rulsif import RuLSIFDensityRatioEstimator
from gpdre.external.kliep import KLIEPDensityRatioEstimator
from gpdre.initializers import KMeans

from gpflow.models import SVGP
from gpflow.kernels import SquaredExponential, Matern52

from sklearn.svm import SVC
from sklearn.datasets import make_moons
# %%

K.set_floatx('float64')
# %%

# shortcuts
tfd = tfp.distributions

# constants
num_samples = 1000

num_features = 2
num_inducing_points = 300
num_seeds = 10

optimizer = "adam"
epochs = 200
batch_size = 100
buffer_size = 1000
jitter = 1e-6

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

ax.set_title("Without importance weighting (uniform weights)")

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

ax.set_title("With importance weighting (exact density ratio)")

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
def metric(X_train, y_train, X_test, y_test, sample_weight=None,
           random_state=None):

    model = SVC(C=1.0, kernel="rbf", gamma="auto", random_state=random_state)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    return model.score(X_test, y_test)
# %%


metric(X_train, y_train, X_val, y_val, random_state=seed)
# %%

metric(X_train, y_train, X_val, y_val,
       sample_weight=r.ratio(X_train).numpy(),
       random_state=seed)
# %%

# properties of the distribution
props = {
    "mean": tfd.Distribution.mean,
    "mode": tfd.Distribution.mode,
    "median": lambda d: d.distribution.quantile(0.5),
    "sample": tfd.Distribution.sample,  # single sample
}
# %%

# kernels = {
#     "squared_exponental": SquaredExponential,
#     "matern52": Matern52
# }
# # %%

# rows = []

# for seed in range(num_seeds):

#     for whiten in [False, True]:

#         for kernel_name, kernel_cls in kernels.items():

#             gpdre = GaussianProcessDensityRatioEstimator(
#                 input_dim=num_features,
#                 kernel_cls=kernel_cls,
#                 num_inducing_points=num_inducing_points,
#                 inducing_index_points_initializer=KMeans(X, seed=seed),
#                 vgp_cls=SVGP,
#                 whiten=whiten,
#                 jitter=jitter,
#                 seed=seed)
#             gpdre.compile(optimizer=optimizer)
#             gpdre.fit(X_val, X_train, epochs=epochs, batch_size=batch_size,
#                       buffer_size=buffer_size)

#             for prop_name, prop in props.items():

#                 r_prop = gpdre.ratio(X_train, convert_to_tensor_fn=prop)
#                 acc = metric(X_train, y_train, X_val, y_val,
#                              sample_weight=r_prop.numpy(), random_state=seed)
#                 rows.append(dict(weight=prop_name, kernel=kernel_name,
#                                  whiten=whiten, acc=acc, seed=seed))
# # %%

# data = pd.DataFrame(rows)
# data
# # %%

# g = sns.catplot(x="kernel", y="acc", hue="weight", col="whiten",
#                 palette="colorblind", dodge=True, alpha=0.6, kind="strip",
#                 data=data)
# %%

rows = []

for seed in range(num_seeds):

    (X_train, y_train), (X_val, y_val) = r.train_test_split(X, y, seed=seed)

    # Uniform
    acc = metric(X_train, y_train, X_val, y_val, random_state=seed)
    rows.append(dict(weight="uniform", acc=acc, seed=seed))

    # Exact
    acc = metric(X_train, y_train, X_val, y_val,
                 sample_weight=r.ratio(X_train).numpy(), random_state=seed)
    rows.append(dict(weight="exact", acc=acc, seed=seed))

    # RuLSIF
    r_rulsif = RuLSIFDensityRatioEstimator(alpha=1e-6)
    r_rulsif.fit(X_val, X_train)
    sample_weight = np.maximum(1e-6, r_rulsif.ratio(X_train))
    acc = metric(X_train, y_train, X_val, y_val,
                 sample_weight=sample_weight, random_state=seed)
    rows.append(dict(weight="rulsif", acc=acc, seed=seed))

    # KLIEP
    r_kliep = KLIEPDensityRatioEstimator(seed=seed)
    r_kliep.fit(X_val, X_train)
    sample_weight = np.maximum(1e-6, r_kliep.ratio(X_train))
    acc = metric(X_train, y_train, X_val, y_val,
                 sample_weight=sample_weight, random_state=seed)
    rows.append(dict(weight="kliep", acc=acc, seed=seed))

    # Logistic Regression (Linear)
    r_linear = MLPDensityRatioEstimator(num_layers=0, num_units=None, seed=seed)
    r_linear.compile(optimizer=optimizer, metrics=["accuracy"])
    r_linear.fit(X_val, X_train, epochs=epochs, batch_size=batch_size)
    sample_weight = np.maximum(1e-6, r_linear.ratio(X_train).numpy())
    acc = metric(X_train, y_train, X_val, y_val,
                 sample_weight=sample_weight, random_state=seed)
    rows.append(dict(weight="logreg_linear", acc=acc, seed=seed))

    # Logistic Regression (MLP)
    r_mlp = MLPDensityRatioEstimator(num_layers=1, num_units=8,
                                     activation="tanh", seed=seed)
    r_mlp.compile(optimizer=optimizer, metrics=["accuracy"])
    r_mlp.fit(X_val, X_train, epochs=epochs, batch_size=batch_size)
    sample_weight = np.maximum(1e-6, r_mlp.ratio(X_train).numpy())
    acc = metric(X_train, y_train, X_val, y_val,
                 sample_weight=sample_weight, random_state=seed)
    rows.append(dict(weight="logreg_deep", acc=acc, seed=seed))

    # Gaussian Processes
    gpdre = GaussianProcessDensityRatioEstimator(
        input_dim=num_features,
        kernel_cls=Matern52,
        num_inducing_points=num_inducing_points,
        inducing_index_points_initializer=KMeans(X, seed=seed),
        vgp_cls=SVGP,
        whiten=True,
        jitter=jitter,
        seed=seed)
    gpdre.compile(optimizer=optimizer)
    gpdre.fit(X_val, X_train, epochs=epochs, batch_size=batch_size,
              buffer_size=buffer_size)

    for prop_name, prop in props.items():

        r_prop = gpdre.ratio(X_train, convert_to_tensor_fn=prop)
        acc = metric(X_train, y_train, X_val, y_val,
                     sample_weight=r_prop.numpy(), random_state=seed)
        rows.append(dict(weight=f"gp_{prop_name}", acc=acc, seed=seed))
# %%

data = pd.DataFrame(rows)
data = data.assign(error=1.0-data.acc)
data
# %%

fig, ax = plt.subplots(figsize=(6, 6))

sns.stripplot(x="error", y="weight", jitter=False, dodge=True,
              data=data, palette="colorblind", alpha=0.4, zorder=1, ax=ax)

sns.pointplot(x="error", y="weight", dodge=0.7,
              data=data, palette="dark", join=False, ci=None, markers='d',
              scale=0.75, ax=ax)

plt.show()
# %%


print(data.groupby("weight")["error"].describe().to_latex(
    columns=["mean", "std"],
    float_format="{:0.3f}".format,
    caption=f"Synthetic problems. Test error across {num_seeds:d} trials.",
    label="tab:synthetic-results",
    formatters={"std": r"($\pm${:0.3f})".format},
    escape=False))
