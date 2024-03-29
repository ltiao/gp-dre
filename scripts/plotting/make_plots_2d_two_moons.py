import sys
import click

import numpy as np
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
import seaborn as sns

from gpdre import DensityRatioMarginals

from sklearn.svm import SVC
from sklearn.datasets import make_moons

from pathlib import Path

# shortcuts
tfd = tfp.distributions

GOLDEN_RATIO = 0.5 * (1 + np.sqrt(5))
WIDTH = 397.48499
OUTPUT_DIR = "logs/figures/"

SEED = 8888
DATASET_SEED = 42
NUM_SAMPLES = 1000


def pt_to_in(x):

    pt_per_in = 72.27
    return x / pt_per_in


def size(width, aspect=GOLDEN_RATIO):

    width_in = pt_to_in(width)
    return (width_in, width_in / aspect)


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

    xmin, xmax = -1.5, 2.5
    ymin, ymax = -1.0, 1.5

    X2, X1 = np.mgrid[ymin:ymax:200j, xmin:xmax:200j]
    X_grid = np.dstack((X1, X2))
    X, y = make_moons(NUM_SAMPLES, noise=0.05, random_state=DATASET_SEED)
    test = tfd.MultivariateNormalDiag(loc=[0.5, 0.25], scale_diag=[0.5, 0.5])
    train = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[0.5, 0.5]),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=[[-1., -0.5], [2., 1.0]], scale_diag=[0.5, 1.5])
    )
    r = DensityRatioMarginals(top=test, bot=train)
    (X_train, y_train), (X_test, y_test) = r.train_test_split(X, y, seed=SEED)

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

    model_uniform_train = SVC(C=1.0, kernel="rbf", gamma="scale", max_iter=-1,
                              probability=True, random_state=SEED)
    model_uniform_train.fit(X_train, y_train)
    model_uniform_test = SVC(C=1.0, kernel="rbf", gamma="scale", max_iter=-1,
                             probability=True, random_state=SEED)
    model_uniform_test.fit(X_test, y_test)
    model_exact_train = SVC(C=1.0, kernel="rbf", gamma="scale", max_iter=-1,
                            probability=True, random_state=SEED)
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
        fig.savefig(output_path.joinpath(f"svc_uniform_{suffix}.{ext}"),
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
        fig.savefig(output_path.joinpath(f"svc_exact_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    # Figure 9
    fig, ax = plt.subplots()

    contours1 = ax.contour(X1, X2, p_grid_uniform_train[..., -1],  linestyles="solid", levels=1, zorder=-1, cmap="Blues")
    contours2 = ax.contour(X1, X2, p_grid_uniform_test[..., -1],  linestyles="dashed", levels=1, zorder=-1, cmap="Oranges")
    contours3 = ax.contour(X1, X2, p_grid_exact_train[..., -1],  linestyles="dotted", levels=1, zorder=-1, cmap="Greens")

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

    ax.clabel(contours_train, fmt="%.2f")
    ax.clabel(contours_test, fmt="%.2f")

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')

    for ext in extension:
        fig.savefig(output_path.joinpath(f"distribution_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
