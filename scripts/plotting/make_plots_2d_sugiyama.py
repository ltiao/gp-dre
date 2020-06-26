import sys
import click

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
import seaborn as sns

from gpdre.benchmarks import SugiyamaKrauledatMuellerDensityRatioMarginals
from gpdre.datasets import make_classification_dataset

from sklearn.linear_model import LogisticRegression

from pathlib import Path

# shortcuts
tfd = tfp.distributions

GOLDEN_RATIO = 0.5 * (1 + np.sqrt(5))
WIDTH = 397.48499
OUTPUT_DIR = "logs/figures/"

SEED = 8888
DATASET_SEED = 2
NUM_TRAIN = 500
NUM_TEST = 500


def pt_to_in(x):

    pt_per_in = 72.27
    return x / pt_per_in


def size(width, aspect=GOLDEN_RATIO):

    width_in = pt_to_in(width)
    return (width_in, width_in / aspect)


def class_posterior(x1, x2):
    return 0.5 * (1 + tf.tanh(x1 - tf.nn.relu(-x2)))


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

    xmin, xmax = -6, 8
    ymin, ymax = -4, 9

    X2, X1 = np.mgrid[ymin:ymax:200j, xmin:xmax:200j]
    X_grid = np.dstack((X1, X2))
    r = SugiyamaKrauledatMuellerDensityRatioMarginals()
    (X_train, y_train), (X_test, y_test) = r.make_covariate_shift_dataset(
      NUM_TEST, NUM_TRAIN, class_posterior_fn=class_posterior, seed=DATASET_SEED)
    X, s = make_classification_dataset(X_test, X_train)

    # Figure 3
    fig, ax = plt.subplots()

    ax.set_title(r"$f(\mathbf{x}) = "
                 r"\log p_{\mathrm{te}}(\mathbf{x}) - "
                 r"\log p_{\mathrm{tr}}(\mathbf{x})$")

    contours = ax.contour(X1, X2, r.logit(X_grid).numpy(), cmap="PuOr")

    fig.colorbar(contours, ax=ax)
    ax.clabel(contours, fmt="%.2f")

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    for ext in extension:
        fig.savefig(output_path.joinpath(f"logit_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    # Figure 4
    fig, ax = plt.subplots()

    ax.set_title(r"$r(\mathbf{x}) = \exp(f(\mathbf{x}))$")

    contours = ax.contour(X1, X2, r.ratio(X_grid).numpy(), cmap="PuOr")

    fig.colorbar(contours, ax=ax)
    ax.clabel(contours, fmt="%.2f")

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    for ext in extension:
        fig.savefig(output_path.joinpath(f"ratio_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    # Figure 5
    fig, ax = plt.subplots()

    ax.set_title(r"$P(s=1|\mathbf{x}) = \sigma(f(\mathbf{x}))$")

    contours = ax.contour(X1, X2, r.prob(X_grid).numpy(), cmap="PuOr")

    ax.scatter(*X.T, c=r.prob(X).numpy(), cmap="PuOr", alpha=0.6)

    fig.colorbar(contours, ax=ax)
    ax.clabel(contours, fmt="%.2f")

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    for ext in extension:
        fig.savefig(output_path.joinpath(f"prob_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    model_uniform_train = LogisticRegression(C=1.0, random_state=SEED)
    model_uniform_train.fit(X_train, y_train)
    model_uniform_test = LogisticRegression(C=1.0, random_state=SEED)
    model_uniform_test.fit(X_test, y_test)
    model_exact_train = LogisticRegression(C=1.0, random_state=SEED)
    model_exact_train.fit(X_train, y_train,
                          sample_weight=r.ratio(X_train).numpy())

    p_grid_uniform_train = model_uniform_train.predict_proba(X_grid.reshape(-1, 2)) \
                                              .reshape(200, 200, 2)
    p_grid_uniform_test = model_uniform_test.predict_proba(X_grid.reshape(-1, 2)) \
                                            .reshape(200, 200, 2)
    p_grid_exact_train = model_exact_train.predict_proba(X_grid.reshape(-1, 2)) \
                                          .reshape(200, 200, 2)

    # Figure 7
    fig, ax = plt.subplots()

    ax.set_title("Without importance sampling (uniform weights)")

    contours = ax.contour(X1, X2, p_grid_uniform_train[..., -1], cmap="RdYlBu")

    fig.colorbar(contours, ax=ax)
    ax.clabel(contours, fmt="%.2f")

    ax.scatter(*X_train.T, c=y_train, cmap="RdYlBu", alpha=0.8, label="train")
    ax.scatter(*X_test.T, marker='x', c=y_test, cmap="RdYlBu", alpha=0.2, label="test")

    ax.legend()

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    for ext in extension:
        fig.savefig(output_path.joinpath(f"logistic_uniform_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    # Figure 8
    fig, ax = plt.subplots()

    ax.set_title("With importance sampling (exact density ratio)")

    contours = ax.contour(X1, X2, p_grid_exact_train[..., -1], cmap="RdYlBu")

    fig.colorbar(contours, ax=ax)
    ax.clabel(contours, fmt="%.2f")

    ax.scatter(*X_train.T, c=y_train, s=r.ratio(X_train).numpy(),
               cmap="RdYlBu", alpha=0.8, label="train")
    ax.scatter(*X_test.T, marker='x', c=y_test,  # s=r.ratio(X_test).numpy(),
               cmap="RdYlBu", alpha=0.2, label="test")

    ax.legend()

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    for ext in extension:
        fig.savefig(output_path.joinpath(f"logistic_exact_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    # Figure 9
    fig, ax = plt.subplots()

    contours1 = ax.contour(X1, X2, p_grid_uniform_train[..., -1],  linestyles="solid", levels=1, zorder=-1, cmap="Blues")
    contours2 = ax.contour(X1, X2, p_grid_uniform_test[..., -1],  linestyles="dashed", levels=1, zorder=-1, cmap="Oranges")
    contours3 = ax.contour(X1, X2, p_grid_exact_train[..., -1],  linestyles="dotted", levels=1, zorder=-1, cmap="Greens")
    contours4 = ax.contour(X1, X2, class_posterior(X1, X2).numpy(),  linestyles="dashdot", levels=1, zorder=-1, cmap="Purples")

    # fig.colorbar(contours, ax=ax)
    # ax.clabel(contours1, fmt="unweighted", fontsize="smaller")
    # ax.clabel(contours2, fmt="test", fontsize="smaller")
    # ax.clabel(contours3, fmt="ideal", fontsize="smaller")

    ax.scatter(*X_train.T, c=y_train, cmap="RdYlBu", alpha=0.8, label="train")
    ax.scatter(*X_test.T, marker='x', c=y_test, cmap="RdYlBu", alpha=0.2, label="test")

    ax.legend()

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    for ext in extension:
        fig.savefig(output_path.joinpath(f"dataset_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    # Figure 2
    fig, ax = plt.subplots()

    # ax.set_title(r"Train $p_{\mathrm{tr}}(\mathbf{x})$ and "
    #              r"test $p_{\mathrm{te}}(\mathbf{x})$ distributions")

    contours_train = ax.contour(X1, X2, r.bot.prob(X_grid), cmap="Oranges")
    contours_test = ax.contour(X1, X2, r.top.prob(X_grid), cmap="Purples",
                               linestyles="--")

    contours = ax.contour(X1, X2, class_posterior(X1, X2).numpy(),
                          levels=1, zorder=-1, linestyles="dashdot",
                          cmap="Purples")

    ax.clabel(contours_train, fmt="%.2f")
    ax.clabel(contours_test, fmt="%.2f")

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    for ext in extension:
        fig.savefig(output_path.joinpath(f"distribution_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
