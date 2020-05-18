"""Console script for zalando_classification."""
import sys
import click

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp

import pandas as pd

from gpdre import GaussianProcessDensityRatioEstimator
from gpdre.benchmarks import SugiyamaKrauledatMuellerDensityRatioMarginals
from gpdre.datasets import make_classification_dataset
from gpdre.base import MLPDensityRatioEstimator
from gpdre.external.rulsif import RuLSIFDensityRatioEstimator
from gpdre.external.kliep import KLIEPDensityRatioEstimator
from gpdre.initializers import KMeans

from gpflow.models import SVGP
from gpflow.kernels import Matern52

from sklearn.linear_model import LogisticRegression

from pathlib import Path

K.set_floatx("float64")

# shortcuts
tfd = tfp.distributions

# sensible defaults
SUMMARY_DIR = "logs/"
SEED = 8888

dataset_seed = 8888

num_features = 2
num_samples = 1000

num_train = 500
num_test = 500

num_inducing_points = 300

optimizer = "adam"
epochs = 1000
batch_size = 100
buffer_size = 1000
jitter = 1e-6

num_seeds = 10

# properties of the distribution
props = {
    "mean": tfd.Distribution.mean,
    # "mode": tfd.Distribution.mode,
    # "median": lambda d: d.distribution.quantile(0.5),
    # "sample": tfd.Distribution.sample,  # single sample
}


def class_posterior(x1, x2):
    return 0.5 * (1 + tf.tanh(x1 - tf.nn.relu(-x2)))


def metric(X_train, y_train, X_test, y_test, sample_weight=None,
           random_state=None):

    model = LogisticRegression(C=1.0, random_state=random_state)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    return model.score(X_test, y_test)


@click.command()
@click.argument("name")
@click.option("--summary-dir", default=SUMMARY_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Summary directory.")
@click.option("-s", "--seed", default=SEED, type=int, help="Random seed")
def main(name, summary_dir, seed):

    summary_path = Path(summary_dir)  # .joinpath(name)
    summary_path.mkdir(parents=True, exist_ok=True)

    r = SugiyamaKrauledatMuellerDensityRatioMarginals()

    rows = []

    for seed in range(num_seeds):

        # (X_train, y_train), (X_test, y_test) = r.train_test_split(X, y, seed=seed)

        (X_train, y_train), (X_test, y_test) = r.make_covariate_shift_dataset(
          num_test, num_train, class_posterior_fn=class_posterior, threshold=0.5,
          seed=seed)
        X, s = make_classification_dataset(X_test, X_train)

        # Uniform
        acc = metric(X_train, y_train, X_test, y_test, random_state=seed)
        rows.append(dict(weight="uniform", acc=acc, seed=seed))

        # Exact
        acc = metric(X_train, y_train, X_test, y_test,
                     sample_weight=r.ratio(X_train).numpy(), random_state=seed)
        rows.append(dict(weight="exact", acc=acc, seed=seed))

        # RuLSIF
        r_rulsif = RuLSIFDensityRatioEstimator(alpha=1e-6)
        r_rulsif.fit(X_test, X_train)
        sample_weight = np.maximum(1e-6, r_rulsif.ratio(X_train))
        acc = metric(X_train, y_train, X_test, y_test,
                     sample_weight=sample_weight, random_state=seed)
        rows.append(dict(weight="rulsif", acc=acc, seed=seed))

        # KLIEP
        r_kliep = KLIEPDensityRatioEstimator(seed=seed)
        r_kliep.fit(X_test, X_train)
        sample_weight = np.maximum(1e-6, r_kliep.ratio(X_train))
        acc = metric(X_train, y_train, X_test, y_test,
                     sample_weight=sample_weight, random_state=seed)
        rows.append(dict(weight="kliep", acc=acc, seed=seed))

        # Logistic Regression (Linear)
        r_linear = MLPDensityRatioEstimator(num_layers=0, num_units=None, seed=seed)
        r_linear.compile(optimizer=optimizer, metrics=["accuracy"])
        r_linear.fit(X_test, X_train, epochs=epochs, batch_size=batch_size)
        sample_weight = np.maximum(1e-6, r_linear.ratio(X_train).numpy())
        acc = metric(X_train, y_train, X_test, y_test,
                     sample_weight=sample_weight, random_state=seed)
        rows.append(dict(weight="logreg_linear", acc=acc, seed=seed))

        # Logistic Regression (MLP)
        r_mlp = MLPDensityRatioEstimator(num_layers=1, num_units=8,
                                         activation="tanh", seed=seed)
        r_mlp.compile(optimizer=optimizer, metrics=["accuracy"])
        r_mlp.fit(X_test, X_train, epochs=epochs, batch_size=batch_size)
        sample_weight = np.maximum(1e-6, r_mlp.ratio(X_train).numpy())
        acc = metric(X_train, y_train, X_test, y_test,
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
        gpdre.fit(X_test, X_train, epochs=epochs, batch_size=batch_size,
                  buffer_size=buffer_size)

        for prop_name, prop in props.items():

            r_prop = gpdre.ratio(X_train, convert_to_tensor_fn=prop)
            acc = metric(X_train, y_train, X_test, y_test,
                         sample_weight=r_prop.numpy(), random_state=seed)
            rows.append(dict(weight=f"gp_{prop_name}", acc=acc, seed=seed))

    data = pd.DataFrame(rows)
    data.to_csv(str(summary_path.joinpath(f"{name}.csv")))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
