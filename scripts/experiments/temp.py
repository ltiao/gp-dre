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
from gpdre.initializers import KMeans

from gpflow.models import SVGP
from gpflow.kernels import SquaredExponential, Matern52

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
num_train = 500
num_test = 500

# sparsity_factors = np.linspace(0.15, 0.5, 3)

sparsity_factor = 0.3
use_ard = True

optimizer = "adam"
# epochs = 1000
batch_size = 100
buffer_size = 1000
jitter = 1e-6

num_seeds = 10

# properties of the distribution
props = {
    "mean": tfd.Distribution.mean,
    "mode": tfd.Distribution.mode,
    "median": lambda d: d.distribution.quantile(0.5),
    "sample": tfd.Distribution.sample,  # single sample
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

        # # Uniform
        # acc = metric(X_train, y_train, X_test, y_test, random_state=seed)
        # rows.append(dict(weight="uniform", acc=acc, seed=seed))

        # # Exact
        # acc = metric(X_train, y_train, X_test, y_test,
        #              sample_weight=r.ratio(X_train).numpy(), random_state=seed)
        # rows.append(dict(weight="exact", acc=acc, seed=seed))

        for epochs in [500, 1000, 1500, 2000]:

            # for sparsity_factor in sparsity_factors:

            num_inducing_points = int(len(X) * sparsity_factor)

            # Gaussian Processes
            gpdre = GaussianProcessDensityRatioEstimator(
                input_dim=num_features,
                kernel_cls=Matern52,
                use_ard=use_ard,
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
                rows.append(dict(weight=prop_name, acc=acc, seed=seed,
                                 sparsity_factor=sparsity_factor,
                                 use_ard=use_ard, epochs=epochs))

    data = pd.DataFrame(rows)
    data.to_csv(str(summary_path.joinpath(f"{name}.csv")))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
