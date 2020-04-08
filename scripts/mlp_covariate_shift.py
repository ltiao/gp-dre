"""Console script for zalando_classification."""
import sys
import click


import pandas as pd

from pathlib import Path

from gpdre.applications.covariate_shift import (
  get_dataset, MLPCovariateShiftAdapter,
  Classification2DCovariateShiftBenchmark,
  TrueCovariateShiftAdapter)
from gpdre.utils import DensityRatio
from gpdre.metrics import normalized_mean_squared_error

# Sensible defaults

# DRE
OPTIMIZER1 = "adam"
EPOCHS1 = 50
BATCH_SIZE = 64

NUM_LAYERS = 1
NUM_UNITS = 8
ACTIVATION = "tanh"

L1_FACTOR = 0.0
L2_FACTOR = 0.0

OPTIMIZER2 = "lbfgs"
EPOCHS2 = 500
PENALTY = "l2"

SUMMARY_DIR = "logs/"

NUM_TRAIN = 500
NUM_TEST = 500

THRESHOLD = 0.5

SEED = 0
DATASET_SEED = 8888


class MLPExperiment:

    def __init__(self, num_layers, num_units, activation, l1_factor, l2_factor,
                 optimizer_auxiliary, epochs_auxiliary, batch_size, optimizer,
                 epochs, penalty, seed=None):

        # Downstream prediction task
        self.classification_problem = Classification2DCovariateShiftBenchmark(
            optimizer=optimizer, epochs=epochs, penalty=penalty, seed=seed)

        # Importance weights
        self.adapter = MLPCovariateShiftAdapter(num_layers, num_units,
                                                activation, l1_factor,
                                                l2_factor, optimizer_auxiliary,
                                                epochs_auxiliary, batch_size,
                                                seed=seed)

        self.adapter_true = TrueCovariateShiftAdapter(
            true_density_ratio=DensityRatio.from_covariate_shift_example())

        self.parameters = dict(num_layers=num_layers, num_units=num_units,
                               activation=activation, l1_factor=l1_factor,
                               l2_factor=l2_factor,
                               optimizer_auxiliary=optimizer_auxiliary,
                               epochs_auxiliary=epochs_auxiliary,
                               batch_size=batch_size, optimizer=optimizer,
                               epochs=epochs, penalty=penalty, seed=seed)

    def get_result(self, X_train, y_train, X_test, y_test):

        importance_weights_true = self.adapter_true.importance_weights(X_train,
                                                                       X_test)
        importance_weights = self.adapter.importance_weights(X_train, X_test)

        auxiliary_accuracy = self.adapter.accuracy(X_train, X_test)
        nmse = normalized_mean_squared_error(importance_weights_true,
                                             importance_weights)
        test_accuracy = self.classification_problem.test_metric(
            train_data=(X_train, y_train), test_data=(X_test, y_test),
            importance_weights=importance_weights)

        results = dict(self.parameters)
        results.update(dict(auxiliary_accuracy=auxiliary_accuracy, nmse=nmse,
                            test_accuracy=test_accuracy))

        return results


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
@click.option("-l", "--num-layers", default=NUM_LAYERS, type=int,
              help="Number of hidden layers.")
@click.option("-u", "--num-units", default=NUM_UNITS, type=int,
              help="Number of hidden units.")
@click.option("--activation", default=ACTIVATION, type=str)
@click.option("--l1-factor", default=L1_FACTOR, type=float,
              help="L1 regularization factor.")
@click.option("--l2-factor", default=L2_FACTOR, type=float,
              help="L2 regularization factor.")
@click.option("--optimizer1", default=OPTIMIZER1,
              help="Optimizer for DRE.")
@click.option("--epochs1", default=EPOCHS1, type=int,
              help="Number of epochs.")
@click.option("-b", "--batch-size", default=BATCH_SIZE, type=int,
              help="Batch size.")
@click.option("--optimizer2", default=OPTIMIZER2,
              help="Optimizer for the downstream prediction task.")
@click.option("--epochs2", default=EPOCHS2, type=int,
              help="Number of epochs for the downstream prediction task.")
@click.option("--dataset-seed", default=DATASET_SEED, type=int)
@click.option("-s", "--seed", default=SEED, type=int, help="Random seed")
def main(name, summary_dir, num_train, num_test, threshold, num_layers,
         num_units, activation, l1_factor, l2_factor, optimizer1, epochs1,
         batch_size, optimizer2, epochs2, dataset_seed, seed):

    mlp_experiment = MLPExperiment(num_layers, num_units, activation,
                                   l1_factor, l2_factor, optimizer1, epochs1,
                                   batch_size, optimizer2, epochs2,
                                   penalty=PENALTY, seed=seed)

    # Get data
    (X_train, y_train), (X_test, y_test) = get_dataset(num_train, num_test,
                                                       threshold=threshold,
                                                       seed=dataset_seed)

    results = mlp_experiment.get_results(X_train, y_train, X_test, y_test)

    click.secho("[Seed {seed:04d}] test accuracy: {test_accuracy:.3f}"
                .format(**results), fg="green")

    # Save results
    summary_path = Path(summary_dir).joinpath(name)
    summary_path.mkdir(parents=True, exist_ok=True)

    data = pd.Series(results)
    data.to_json(str(summary_path.joinpath(f"{seed:04d}.json")))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
