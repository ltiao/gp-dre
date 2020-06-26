"""Console script for zalando_classification."""
import sys
import click

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from gpdre.utils import DensityRatio
from gpdre.applications.covariate_shift import (
  MLPCovariateShiftAdapter, ExactCovariateShiftAdapter)
from gpdre.applications.covariate_shift.benchmarks import CovariateShiftBenchmark
from gpdre.metrics import normalized_mean_squared_error

from sklearn.linear_model import Ridge
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state
from sklearn.preprocessing import MinMaxScaler

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


class RegressionCovariateShiftBenchmark(CovariateShiftBenchmark):

    def __init__(self, optimizer="auto", epochs=500, seed=None):

        self.model = Ridge(alpha=1.0, solver=optimizer, max_iter=epochs,
                           random_state=seed)

    def test_metric(self, train_data, test_data, importance_weights=None):

        X_train, y_train = train_data
        X_test, y_test = test_data

        self.model.fit(X_train, y_train, sample_weight=importance_weights)
        y_pred = self.model.predict(X_test)

        return mean_squared_error(y_test, y_pred, squared=False)


class MLPExperiment:

    def __init__(self, num_layers, num_units, activation, l1_factor, l2_factor,
                 optimizer_auxiliary, epochs_auxiliary, batch_size, optimizer,
                 epochs, seed=None):

        # Downstream prediction task
        self.benchmark = RegressionCovariateShiftBenchmark(
            optimizer=optimizer, epochs=epochs, seed=seed)

        # Importance weights
        self.adapter = MLPCovariateShiftAdapter(num_layers, num_units,
                                                activation, l1_factor,
                                                l2_factor, optimizer_auxiliary,
                                                epochs_auxiliary, batch_size,
                                                seed=seed)

        # self.adapter_true = ExactCovariateShiftAdapter(
        #     exact_density_ratio=DensityRatio.from_covariate_shift_example())

        self.parameters = dict(num_layers=num_layers, num_units=num_units,
                               activation=activation, l1_factor=l1_factor,
                               l2_factor=l2_factor,
                               optimizer_auxiliary=optimizer_auxiliary,
                               epochs_auxiliary=epochs_auxiliary,
                               batch_size=batch_size, optimizer=optimizer,
                               epochs=epochs, seed=seed)

    def get_result(self, X_train, y_train, X_test, y_test):

        # importance_weights_true = self.adapter_true.importance_weights(X_train,
        #                                                                X_test)
        importance_weights = self.adapter.importance_weights(X_train, X_test)

        auxiliary_accuracy = self.adapter.accuracy(X_train, X_test)
        # nmse = normalized_mean_squared_error(importance_weights_true,
        #                                      importance_weights)
        test_accuracy = self.classification_problem.test_metric(
            train_data=(X_train, y_train), test_data=(X_test, y_test),
            importance_weights=importance_weights)

        results = dict(self.parameters)
        results.update(dict(auxiliary_accuracy=auxiliary_accuracy,  # nmse=nmse,
                            test_accuracy=test_accuracy))

        return results


class CortesDensityRatio(DensityRatio):

    def __init__(self, input_dim, seed=None):

        self.rng = check_random_state(seed)
        self.w = self.rng.uniform(low=-1.0, high=1.0, size=input_dim)

    def logit(self, X):

        X_bar = np.mean(X, axis=-1, keepdims=True)
        X_tilde = X - X_bar

        u = np.dot(X_tilde, self.w)

        return - 4.0 * u / np.std(u)

    def train_test_split(self, X, y):

        mask_test = self.rng.binomial(n=1, p=self.prob(X).numpy()).astype(bool)
        mask_train = ~mask_test

        return (X[mask_train], y[mask_train]), (X[mask_test], y[mask_test])


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

    num_layers = 2
    num_units = 16
    activation = "relu"
    l1_factor = 0.0
    l2_factor = 0.0
    optimizer_auxiliary = "adam"
    epochs_auxiliary = 100
    batch_size = 64
    optimizer = "auto"
    epochs = 1000

    num_splits = 20
    num_seeds = 15

    california_housing = fetch_california_housing(data_home="../datasets")
    # abalone = fetch_openml(data_id=572, data_home="../datasets")

    results_list = []

    for split in range(num_splits):

        density_ratio = CortesDensityRatio(input_dim=8, seed=split)

        (X_train, y_train), (X_test, y_test) = density_ratio.train_test_split(
            # X=MinMaxScaler().fit_transform(california_housing.data),
            X=california_housing.data, y=california_housing.target)

        num_train = len(X_train)
        num_test = len(X_test)

        # if num_test / (num_train + num_test)
        click.secho(f"[split {split:03d}] "
                    f"num_train {num_train:d}, "
                    f"num_test {num_test:d}", fg="green")

        adapter_exact = ExactCovariateShiftAdapter(
            exact_density_ratio=density_ratio)
        importance_weights_exact = adapter_exact.importance_weights(X_train,
                                                                    X_test)
        for seed in range(num_seeds):

            # Downstream prediction task
            benchmark = RegressionCovariateShiftBenchmark(
                optimizer=optimizer, epochs=epochs, seed=seed)

            test_rmse_uniform = benchmark.test_metric(
                train_data=(X_train, y_train), test_data=(X_test, y_test),
                importance_weights=None)

            test_rmse_exact = benchmark.test_metric(
                train_data=(X_train, y_train), test_data=(X_test, y_test),
                importance_weights=importance_weights_exact)

            # for activation in ["relu", "tanh"]:

            #     for num_layers in range(5):

            #         for num_units_log2 in range(3, 7):

            #             num_units = 1 << num_units_log2

            # Importance weights
            adapter = MLPCovariateShiftAdapter(num_layers, num_units,
                                               activation, l1_factor,
                                               l2_factor, optimizer_auxiliary,
                                               epochs_auxiliary, batch_size,
                                               seed=seed)
            importance_weights_mlp = adapter.importance_weights(X_train, X_test)

            auxiliary_accuracy = adapter.accuracy(X_train, X_test)
            nmse = normalized_mean_squared_error(importance_weights_exact,
                                                 importance_weights_mlp)

            test_rmse_mlp = benchmark.test_metric(
                train_data=(X_train, y_train), test_data=(X_test, y_test),
                importance_weights=importance_weights_mlp)

            result_dict = dict(auxiliary_accuracy=auxiliary_accuracy,
                               nmse=nmse,
                               test_rmse_uniform=test_rmse_uniform,
                               test_rmse_mlp=test_rmse_mlp,
                               test_rmse_exact=test_rmse_exact,
                               num_layers=num_layers, num_units=num_units,
                               activation=activation, l1_factor=l1_factor,
                               l2_factor=l2_factor,
                               optimizer_auxiliary=optimizer_auxiliary,
                               epochs_auxiliary=epochs_auxiliary,
                               batch_size=batch_size, optimizer=optimizer,
                               epochs=epochs, seed=seed, split=split,
                               num_train=num_train, num_test=num_test)
            results_list.append(result_dict)

            click.secho("[split {split:03d}, seed {seed:04d}] "
                        "test rmse (uniform): {test_rmse_uniform:.3f}, "
                        "test rmse (exact): {test_rmse_exact:.3f}, "
                        "test rmse (mlp): {test_rmse_mlp:.3f}, "
                        "auxiliary accuracy={auxiliary_accuracy:.3f}, "
                        "nmse={nmse:.0e}".format(**result_dict), fg="green")

    # Save results
    results_data = pd.DataFrame(results_list)
    results_data.to_csv(str(summary_path.joinpath(f"{dataset_seed:04d}.csv")))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
