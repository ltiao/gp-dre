"""Console script for zalando_classification."""
import sys
import click

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from gpdre.applications.covariate_shift import get_dataset
from mlp_covariate_shift import MLPExperiment

# Sensible defaults
SUMMARY_DIR = "logs/"

NUM_TRAIN = 500
NUM_TEST = 500

THRESHOLD = 0.5
PENALTY = "l2"

DATASET_SEED = 8888

GOLDEN_RATIO = 0.5 * (1 + np.sqrt(5))
WIDTH = 10.0


def golden_size(width):
    return (width, width / GOLDEN_RATIO)


def g(experiment, summary_path, results, X_train, y_train, figsize=(2*WIDTH, WIDTH / GOLDEN_RATIO),
      fmt_fn="{activation}.{num_layers:02d}.{num_units:02d}.{seed:04d}.png".format):

    ### Figure 1 ###
    def weights_pcolormesh(x1, x2, weights, title, ax):

        ax.set_title(title)

        contours = ax.pcolormesh(x1, x2, weights, vmax=1e+1, cmap="Blues")

        fig.colorbar(contours, extend="max", ax=ax)

        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')

    x2, x1 = np.mgrid[-4:9:50j, -6:8:50j]
    x_grid = np.dstack([x1, x2])
    x_grid_flat = x_grid.reshape(-1, 2)

    y_pred = experiment.classification_problem.model.predict_proba(x_grid_flat)[..., -1] \
                                                    .reshape(50, 50)

    r_true = experiment.adapter_true.true_density_ratio(x_grid).numpy()
    r_pred = experiment.adapter.estimator(x_grid).numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=figsize)

    # ground truth
    weights_pcolormesh(x1, x2, r_true, title="true", ax=ax1)
    # predictions
    weights_pcolormesh(x1, x2, r_pred, title="predicted", ax=ax2)

    fig.suptitle("test accuracy={test_accuracy:.3f} - "
                 "auxiliary accuracy={auxiliary_accuracy:.3f} - "
                 "nmse={nmse:.0e}".format(**results))

    fig.savefig(summary_path.joinpath("pcolor." + fmt_fn(**results)))

    ### Figure 2 ###
    def weights_scatter(X_train, y_train, weights, title, ax):

        ax.set_title(title)

        ax.scatter(*X_train.T, c=y_train, s=1e2*weights,
                   zorder=2, cmap="coolwarm", alpha=0.6)

        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')

    r_true = experiment.adapter_true.true_density_ratio(X_train).numpy()
    r_pred = experiment.adapter.estimator(X_train).numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=figsize)

    # ground truth
    weights_scatter(X_train, y_train, r_true, title="true", ax=ax1)
    # predictions
    weights_scatter(X_train, y_train, r_pred, title="predicted", ax=ax2)

    contours = ax2.contour(x1, x2, y_pred, zorder=1, cmap="coolwarm")
    fig.colorbar(contours, ax=ax2)

    fig.suptitle("test accuracy={test_accuracy:.3f} - "
                 "auxiliary accuracy={auxiliary_accuracy:.3f} - "
                 "nmse={nmse:.0e}".format(**results))

    fig.savefig(summary_path.joinpath("scatter." + fmt_fn(**results)))


@click.command()
@click.argument("name")
@click.option("--summary-dir", default=SUMMARY_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Summary directory.")
@click.option("--num-train", default=NUM_TRAIN, type=int,
              help="Number of training samples")
@click.option("--num-test", default=NUM_TEST, type=int,
              help="Number of test samples")
@click.option("--threshold", default=THRESHOLD, type=float, help="Threshold")
@click.option("--dataset-seed", default=DATASET_SEED, type=int)
def main(name, summary_dir, num_train, num_test, threshold, dataset_seed):

    summary_path = Path(summary_dir).joinpath(name)
    summary_path.mkdir(parents=True, exist_ok=True)

    # Get data
    (X_train, y_train), (X_test, y_test) = get_dataset(num_train, num_test,
                                                       threshold=threshold,
                                                       seed=dataset_seed)

    # num_layers = 2
    # num_units = 16
    activation = "relu"
    l1_factor = 0.0
    l2_factor = 0.0
    optimizer1 = "adam"
    epochs1 = 250
    batch_size = 64
    optimizer2 = "lbfgs"
    epochs2 = 500

    results_list = []

    for seed in range(25):

        for activation in ["relu", "tanh"]:

            for num_layers in range(5):

                for num_units_log2 in range(3, 7):

                    num_units = 1 << num_units_log2

                    mlp_experiment = MLPExperiment(num_layers, num_units,
                                                   activation, l1_factor,
                                                   l2_factor, optimizer1,
                                                   epochs1, batch_size,
                                                   optimizer2, epochs2,
                                                   penalty=PENALTY, seed=seed)

                    results = mlp_experiment.get_result(X_train, y_train,
                                                        X_test, y_test)
                    results_list.append(results)

                    g(mlp_experiment, summary_path, results, X_train, y_train)

                    click.secho("[Seed {seed:04d}] test accuracy: {test_accuracy:.3f}"
                                .format(**results), fg="green")

    # Save results
    results_data = pd.DataFrame(results_list)
    results_data.to_csv(str(summary_path.joinpath(f"{dataset_seed:04d}.csv")))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
