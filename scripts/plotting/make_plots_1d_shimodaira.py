import sys
import click

import numpy as np
import pandas as pd
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
import seaborn as sns

from gpdre import DensityRatioMarginals, GaussianProcessDensityRatioEstimator
from gpdre.datasets import make_classification_dataset
from gpdre.plotting import fill_between_stddev

from gpflow.models import VGP
from gpflow.kernels import Matern52
from gpflow.optimizers import Scipy

from sklearn.linear_model import LinearRegression

from pathlib import Path

# shortcuts
tfd = tfp.distributions

GOLDEN_RATIO = 0.5 * (1 + np.sqrt(5))
WIDTH = 397.48499
OUTPUT_DIR = "logs/figures/"

SEED = 8888
DATASET_SEED = 24
NUM_TRAIN = 100
NUM_TEST = 100

JITTER = 1e-6

num_samples = 50
kernel_cls = Matern52
optimizer = Scipy()


def pt_to_in(x):

    pt_per_in = 72.27
    return x / pt_per_in


def size(width, aspect=GOLDEN_RATIO):

    width_in = pt_to_in(width)
    return (width_in, width_in / aspect)


def poly(x):
    return - x + x**3


@click.command()
@click.argument("name")
@click.option('--width', '-w', type=float, default=WIDTH)
@click.option('--aspect', '-a', type=float, default=GOLDEN_RATIO)
@click.option('--extension', '-e', multiple=True, default=["png"])
@click.option("--output-dir", default=OUTPUT_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(name, width, aspect, extension, output_dir):

    figsize = size(width, aspect)
    suffix = f"{width:.0f}x{width/aspect:.0f}"

    # preamble
    rc = {
        "figure.figsize": figsize,
        "font.serif": ["Times New Roman"],
        "text.usetex": True,
    }

    sns.set(context="paper",
            style="ticks",
            palette="colorblind",
            font="serif",
            rc=rc)

    output_path = Path(output_dir).joinpath(name)
    output_path.mkdir(parents=True, exist_ok=True)
    # /preamble

    xmin, xmax = -1.5, 2.0
    num_features = 1
    num_index_points = 512
    y_num_points = 512

    X_grid = np.linspace(xmin, xmax, num_index_points).reshape(-1, num_features)

    test = tfd.Normal(loc=0.0, scale=0.3)
    train = tfd.Normal(loc=0.5, scale=0.5)
    r = DensityRatioMarginals(top=test, bot=train)
    (X_train, y_train), (X_test, y_test) = r.make_regression_dataset(
        NUM_TEST, NUM_TRAIN, latent_fn=poly, noise_scale=0.3, seed=DATASET_SEED)
    X, s = make_classification_dataset(X_test, X_train)

    # Figure 2
    fig, ax = plt.subplots()

    ax.plot(X_grid, r.bot.prob(X_grid), label=r"$p_{\mathrm{tr}}(\mathbf{x})$")
    ax.plot(X_grid, r.top.prob(X_grid), label=r"$p_{\mathrm{te}}(\mathbf{x})$")

    ax.set_xlabel('$x$')
    ax.set_ylabel('density')

    ax.legend()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"distribution_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    # Figure 3
    fig, ax = plt.subplots()

    ax.plot(X_grid, r.logit(X_grid), c='k',
            label=r"$f(x) = \log p(x) - \log q(x)$")

    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')

    ax.legend()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"logit_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    # Figure 4
    fig, ax = plt.subplots()

    ax.plot(X_grid, r.ratio(X_grid), c='k', label=r"$r(x) = \exp{f(x)}$")

    ax.set_xlabel('$x$')
    ax.set_ylabel('$r(x)$')

    ax.legend()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"ratio_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    model_uniform_train = LinearRegression().fit(X_train, y_train)
    model_uniform_test = LinearRegression().fit(X_test, y_test)
    model_exact_train = LinearRegression().fit(X_train, y_train,
                                               sample_weight=r.ratio(X_train) \
                                                              .numpy().squeeze())

    frames = []
    frames.append(pd.DataFrame(dict(name="uniform train",
                                    x=np.squeeze(X_grid, axis=-1),
                                    y=model_uniform_train.predict(X_grid))))
    frames.append(pd.DataFrame(dict(name="uniform test",
                                    x=np.squeeze(X_grid, axis=-1),
                                    y=model_uniform_test.predict(X_grid))))
    frames.append(pd.DataFrame(dict(name="exact train",
                                    x=np.squeeze(X_grid, axis=-1),
                                    y=model_exact_train.predict(X_grid))))
    frames.append(pd.DataFrame(dict(name="true",
                                    x=np.squeeze(X_grid, axis=-1),
                                    y=np.squeeze(poly(X_grid), axis=-1))))
    data = pd.concat(frames, axis="index", ignore_index=True, sort=True)

    # Figure 7
    fig, ax = plt.subplots()

    sns.lineplot(x='x', y='y', hue="name", style="name", data=data, ax=ax)

    ax.scatter(X_train, y_train, alpha=0.6, label="train")
    ax.scatter(X_test, y_test, marker='x', alpha=0.6, label="test")

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-1.0, 1.5)

    ax.legend(loc="upper left")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[1:], labels[1:], title=None,
              handletextpad=0.5, columnspacing=1,
              ncol=2, frameon=True, loc="upper left")

    for ext in extension:
        fig.savefig(output_path.joinpath(f"dataset_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    gpdre = GaussianProcessDensityRatioEstimator(input_dim=num_features,
                                                 kernel_cls=kernel_cls,
                                                 vgp_cls=VGP,
                                                 jitter=JITTER,
                                                 seed=SEED)
    gpdre.compile(optimizer=optimizer)
    gpdre.fit(X_test, X_train)

    log_ratio_mean = gpdre.logit(X_grid, convert_to_tensor_fn=tfd.Distribution.mean)
    log_ratio_stddev = gpdre.logit(X_grid, convert_to_tensor_fn=tfd.Distribution.stddev)

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

    for ext in extension:
        fig.savefig(output_path.joinpath(f"logit_posterior_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    ratio = gpdre.ratio_distribution(X_grid)

    fig, ax = plt.subplots()

    ax.plot(X_grid, r(X_grid), c='k', label=r"$r(x) = \exp{f(x)}$")

    ax.plot(X_grid, ratio.distribution.quantile(0.5),
            c="tab:blue", label="median")

    ax.fill_between(X_grid.squeeze(),
                    ratio.distribution.quantile(0.25),
                    ratio.distribution.quantile(0.75),
                    alpha=0.1, label="interquartile range")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$r(x)$")

    ax.legend()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"ratio_posterior_quantiles_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    fig, ax = plt.subplots()

    ax.plot(X_grid, r(X_grid), c='k', label=r"$r(x) = \exp{f(x)}$")
    ax.plot(X_grid, ratio.distribution.quantile(0.5),
            c="tab:blue", label="median")
    ax.fill_between(X_grid.squeeze(),
                    ratio.distribution.quantile(0.25),
                    ratio.distribution.quantile(0.75),
                    alpha=0.1, label="interquartile range")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$r(x)$")

    ax.set_ylim(0.0, 7.5)

    ax.legend()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"ratio_posterior_quantiles_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    ratio_marginal = gpdre.ratio_distribution(X_grid,
                                              reinterpreted_batch_ndims=None)

    Y_grid = np.linspace(0.0, 4.0, y_num_points).reshape(-1, 1)
    x, y = np.meshgrid(X_grid, Y_grid)

    fig, ax = plt.subplots()

    contours = ax.pcolormesh(x, y, ratio_marginal.prob(Y_grid),
                             vmax=1.0, cmap="Blues")

    ax.plot(X_grid, r(X_grid), c='k', label=r"$r(x) = \exp{f(x)}$")

    fig.colorbar(contours, extend="max", ax=ax)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$r(x)$")

    ax.legend()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"ratio_posterior_density_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    Y_grid = np.linspace(0.0, 7.5, y_num_points).reshape(-1, 1)
    x, y = np.meshgrid(X_grid, Y_grid)

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

    ax.set_ylim(0.0, 7.5)

    ax.legend()

    for ext in extension:
        fig.savefig(output_path.joinpath(f"ratio_posterior_mode_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    # Figure 5
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

    for ext in extension:
        fig.savefig(output_path.joinpath(f"prob_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
