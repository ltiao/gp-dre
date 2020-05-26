"""Console script for zalando_classification."""
import sys
import click
import h5py

import numpy as np
import tensorflow_probability as tfp

import pandas as pd

from gpdre import GaussianProcessDensityRatioEstimator
from gpdre.base import MLPDensityRatioEstimator, LogisticRegressionDensityRatioEstimator
from gpdre.external.rulsif import RuLSIFDensityRatioEstimator
from gpdre.external.kliep import KLIEPDensityRatioEstimator
from gpdre.external.kmm import KMMDensityRatioEstimator
from gpdre.datasets import make_classification_dataset
from gpdre.metrics import normalized_mean_squared_error
from gpdre.initializers import KMeans

from gpflow.models import SVGP
from gpflow.kernels import SquaredExponential, Matern52

from sklearn.kernel_ridge import KernelRidge

from collections import defaultdict
from scipy.io import loadmat
from pathlib import Path

# shortcuts
tfd = tfp.distributions

# Sensible defaults
num_inducing_points = 300

optimizer = "adam"
epochs = 1000
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


SUMMARY_DIR = "logs/"
SEED = 0


def get_splits(mat, key):

    return list(mat.get(key).squeeze(axis=-1))


def get_data_path(name, data_home="datasets/tmp", as_str=True):

    path = Path(data_home).joinpath(f"{name}_KRR_data.mat")

    if as_str:
        return str(path)

    return path


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

        data_path = get_data_path(name)
        mat = loadmat(data_path)

        X_trains = get_splits(mat, key="X")
        y_trains = get_splits(mat, key="Y")

        X_tests = get_splits(mat, key="Xtest")
        y_tests = get_splits(mat, key="Ytest")

        weights = get_splits(mat, key="ideal")

        for split, (X_train, Y_train, X_test, Y_test, weight) in \
                enumerate(zip(X_trains, y_trains, X_tests, y_tests, weights)):

            X, s = make_classification_dataset(X_test, X_train)
            num_features = X.shape[-1]

            y_train = Y_train.squeeze(axis=-1)
            y_test = Y_test.squeeze(axis=-1)
            sample_weight = weight.squeeze(axis=-1)

            error = regression_metric(X_train, y_train, X_test, y_test)
            rows.append(dict(weight="uniform", name=name, split=split, error=error))

            error = regression_metric(X_train, y_train, X_test, y_test,
                                      sample_weight=sample_weight)
            rows.append(dict(weight="exact", name=name, split=split, error=error))

            # RuLSIF
            r_rulsif = RuLSIFDensityRatioEstimator(alpha=1e-6)
            r_rulsif.fit(X_test, X_train)
            sample_weight = np.maximum(1e-6, r_rulsif.ratio(X_train))
            error = regression_metric(X_train, y_train, X_test, y_test,
                                      sample_weight=sample_weight)
            rows.append(dict(weight="rulsif", name=name, split=split, error=error))

            # # KLIEP
            # r_kliep = KLIEPDensityRatioEstimator(seed=seed)
            # r_kliep.fit(X_test, X_train)
            # sample_weight = np.maximum(1e-6, r_kliep.ratio(X_train))
            # error = regression_metric(X_train, y_train, X_test, y_test,
            #                           sample_weight=sample_weight)
            # rows.append(dict(weight="kliep", name=name, split=split, error=error))

            # KMM
            r_kmm = KMMDensityRatioEstimator(B=1000.0)
            r_kmm.fit(X_test, X_train)
            sample_weight = np.maximum(1e-6, r_kmm.ratio(X_train))
            error = regression_metric(X_train, y_train, X_test, y_test,
                                      sample_weight=sample_weight)
            rows.append(dict(weight="kmm", name=name, split=split, error=error))

            # Logistic Regression (Linear)
            r_logreg = LogisticRegressionDensityRatioEstimator(C=1.0, seed=seed)
            r_logreg.fit(X_test, X_train)
            sample_weight = np.maximum(1e-6, r_logreg.ratio(X_train).numpy())
            error = regression_metric(X_train, y_train, X_test, y_test,
                                      sample_weight=sample_weight)
            rows.append(dict(weight="logreg", name=name, split=split, error=error))

            # Logistic Regression (MLP)
            r_mlp = MLPDensityRatioEstimator(num_layers=2, num_units=32,
                                             activation="relu", seed=seed)
            r_mlp.compile(optimizer=optimizer, metrics=["accuracy"])
            r_mlp.fit(X_test, X_train, epochs=epochs, batch_size=batch_size)
            sample_weight = np.maximum(1e-6, r_mlp.ratio(X_train).numpy())
            error = regression_metric(X_train, y_train, X_test, y_test,
                                      sample_weight=sample_weight)
            rows.append(dict(weight="mlp", name=name, split=split, error=error))

            # Gaussian Processes
            gpdre = GaussianProcessDensityRatioEstimator(
                input_dim=num_features,
                kernel_cls=Matern52,
                num_inducing_points=num_inducing_points,
                inducing_index_points_initializer=KMeans(X, seed=split),
                vgp_cls=SVGP,
                whiten=True,
                jitter=jitter,
                seed=split)
            gpdre.compile(optimizer=optimizer)
            gpdre.fit(X_test, X_train, epochs=epochs, batch_size=batch_size,
                      buffer_size=buffer_size)

            for prop_name, prop in props.items():

                r_prop = gpdre.ratio(X_train, convert_to_tensor_fn=prop)

                error = regression_metric(X_train, y_train, X_test, y_test,
                                          sample_weight=r_prop.numpy())
                rows.append(dict(weight=prop_name, name=name,
                                 split=split, error=error))

    data = pd.DataFrame(rows)
    data.to_csv(str(summary_path.joinpath(f"{myname}.csv")))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
