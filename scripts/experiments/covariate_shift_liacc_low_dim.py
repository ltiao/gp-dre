"""Console script for zalando_classification."""
import sys
import click

import tensorflow_probability as tfp
import pandas as pd

from gpdre import GaussianProcessDensityRatioEstimator
from gpdre.datasets import make_classification_dataset
from gpdre.metrics import normalized_mean_squared_error
from gpdre.initializers import KMeans

from gpflow.models import SVGP
from gpflow.kernels import SquaredExponential, Matern52

from sklearn.kernel_ridge import KernelRidge

from scipy.io import loadmat
from pathlib import Path

from ..utils import get_path, get_splits

# shortcuts
tfd = tfp.distributions

# Sensible defaults
num_inducing_points = 300

optimizer = "adam"
epochs = 800
batch_size = 100
buffer_size = 1000
jitter = 1e-6

# properties of the distribution
props = {
    "mean": tfd.Distribution.mean,
    "mode": tfd.Distribution.mode,
    "median": lambda d: d.distribution.quantile(0.5),
    "sample": tfd.Distribution.sample,  # single sample
}

kernels = {
    "sqr_exp": SquaredExponential,
    "matern52": Matern52
}


SUMMARY_DIR = "logs/"
SEED = 0


def regression_metric(X_train, y_train, X_test, y_test, sample_weight=None):

    input_dim = X_train.shape[-1]
    gamma = 1.0 / input_dim  # gamma = 1/D <=> sigma = sqrt(D/2)

    model = KernelRidge(kernel="rbf", gamma=gamma)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(X_test)

    return normalized_mean_squared_error(y_test, y_pred)


@click.command()
@click.argument("myname")
@click.option("--summary-dir", default=SUMMARY_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Summary directory.")
@click.option("-s", "--seed", default=SEED, type=int, help="Random seed")
def main(myname, summary_dir, seed):

    summary_path = Path(summary_dir).joinpath("liacc")
    summary_path.mkdir(parents=True, exist_ok=True)

    names = ["abalone", "ailerons", "bank32", "bank8", "cali", "cpuact",
             "elevators", "puma8"]

    rows = []

    for name in names:

        output_path = summary_path.joinpath(f"{name}")
        output_path.mkdir(parents=True, exist_ok=True)

        data_path = get_path(name, kind="data", data_home="results/")
        data_mat = loadmat(data_path)

        X_trains = get_splits(data_mat, key="X")
        Y_trains = get_splits(data_mat, key="Y")

        X_tests = get_splits(data_mat, key="Xtest")
        Y_tests = get_splits(data_mat, key="Ytest")

        sample_weights = get_splits(data_mat, key="ideal")

        result_path = get_path(name, kind="results_CV", data_home="results/20200530/")
        results_mat = loadmat(result_path)

        projs = get_splits(results_mat, key="all_W")

        for split, (X_train, Y_train, X_test, Y_test, proj, sample_weight) in \
                enumerate(zip(X_trains, Y_trains, X_tests, Y_tests, projs, sample_weights)):

            y_train = Y_train.squeeze(axis=-1)
            y_test = Y_test.squeeze(axis=-1)

            # X, _ = make_classification_dataset(X_test, X_train)
            # num_features = X.shape[-1]

            X_train_low = X_train.dot(proj)
            X_test_low = X_test.dot(proj)
            X_low, _ = make_classification_dataset(X_test_low, X_train_low)
            num_features_low = X_low.shape[-1]

            error = regression_metric(X_train_low, y_train, X_test_low, y_test)
            rows.append(dict(weight="uniform", name=name, split=split,
                             error=error, projection="low"))

            error = regression_metric(X_train_low, y_train, X_test_low, y_test,
                                      sample_weight=sample_weight.squeeze(axis=-1))
            rows.append(dict(weight="exact", name=name, split=split,
                             error=error, projection="low"))

            # # Gaussian Processes (full-dimensional)
            # gpdre = GaussianProcessDensityRatioEstimator(
            #     input_dim=num_features, kernel_cls=kernels["sqr_exp"],
            #     num_inducing_points=num_inducing_points,
            #     inducing_index_points_initializer=KMeans(X, seed=split),
            #     vgp_cls=SVGP, whiten=True, jitter=jitter, seed=split)
            # gpdre.compile(optimizer=optimizer)
            # gpdre.fit(X_test, X_train,
            #           epochs=epochs,
            #           batch_size=batch_size,
            #           buffer_size=buffer_size)

            # for prop_name, prop in props.items():
            #     r_prop = gpdre.ratio(X_train, convert_to_tensor_fn=prop)
            #     error = regression_metric(X_train, y_train, X_test, y_test,
            #                               sample_weight=r_prop.numpy())
            #     rows.append(dict(name=name, split=split, error=error,
            #                      projection="none", kernel_name="sqr_exp",
            #                      use_ard=True, epochs=epochs, weight=prop_name,
            #                      num_inducing_points=num_inducing_points,
            #                      num_features=num_features))

            # Gaussian Processes (low-dimensional)
            gpdre = GaussianProcessDensityRatioEstimator(
                input_dim=num_features_low, kernel_cls=kernels["sqr_exp"],
                num_inducing_points=num_inducing_points,
                inducing_index_points_initializer=KMeans(X_low, seed=split),
                vgp_cls=SVGP, whiten=True, jitter=jitter, seed=split)
            gpdre.compile(optimizer=optimizer)
            gpdre.fit(X_test_low, X_train_low,
                      epochs=epochs,
                      batch_size=batch_size,
                      buffer_size=buffer_size)

            for prop_name, prop in props.items():
                r_prop = gpdre.ratio(X_train_low, convert_to_tensor_fn=prop)
                error = regression_metric(X_train_low, y_train, X_test_low, y_test,
                                          sample_weight=r_prop.numpy())
                rows.append(dict(name=name, split=split, error=error,
                                 projection="low", kernel_name="sqr_exp",
                                 use_ard=True, epochs=epochs, weight=prop_name,
                                 num_inducing_points=num_inducing_points,
                                 num_features=num_features_low))

            data = pd.DataFrame(rows)
            data.to_csv(str(summary_path.joinpath(f"{myname}.csv")))

    # data = pd.DataFrame(rows)
    # data.to_csv(str(summary_path.joinpath(f"{myname}.csv")))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
