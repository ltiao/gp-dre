"""Console script for zalando_classification."""
import sys
import click

import numpy as np
import tensorflow.keras.backend as K
import tensorflow_probability as tfp

import pandas as pd

from gpdre import DensityRatioMarginals, GaussianProcessDensityRatioEstimator
from gpdre.base import MLPDensityRatioEstimator
from gpdre.external.rulsif import RuLSIFDensityRatioEstimator
from gpdre.external.kliep import KLIEPDensityRatioEstimator
from gpdre.metrics import normalized_mean_squared_error

from gpflow.models import VGP
from gpflow.kernels import Matern52
from gpflow.optimizers import Scipy

from sklearn.linear_model import LinearRegression

from pathlib import Path

K.set_floatx("float64")

# shortcuts
tfd = tfp.distributions

# sensible defaults
SUMMARY_DIR = "logs/"
SEED = 8888

dataset_seed = 8888

num_features = 1

num_train = 100
num_test = 100

kernel_cls = Matern52
optimizer = Scipy()

jitter = 1e-6

num_seeds = 100

# properties of the distribution
props = {
    "mean": tfd.Distribution.mean,
    "mode": tfd.Distribution.mode,
    # "median": lambda d: d.distribution.quantile(0.5),
    # "sample": tfd.Distribution.sample,  # single sample
}


def poly(x):
    return - x + x**3


def metric(X_train, y_train, X_test, y_test, sample_weight=None,
           random_state=None):

    print(sample_weight)

    model = LinearRegression().fit(X_train, y_train,
                                   sample_weight=sample_weight)

    y_pred = model.predict(X_test)

    return normalized_mean_squared_error(y_test, y_pred)


@click.command()
@click.argument("name")
@click.option("--summary-dir", default=SUMMARY_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Summary directory.")
@click.option("-s", "--seed", default=SEED, type=int, help="Random seed")
def main(name, summary_dir, seed):

    summary_path = Path(summary_dir).joinpath("shimodaira")
    summary_path.mkdir(parents=True, exist_ok=True)

    test = tfd.Normal(loc=0.0, scale=0.3)
    train = tfd.Normal(loc=0.5, scale=0.5)
    r = DensityRatioMarginals(top=test, bot=train)

    rows = []

    for seed in range(num_seeds):

        (X_train, y_train), (X_test, y_test) = r.make_regression_dataset(
            num_test, num_train, latent_fn=poly,
            noise_scale=0.3, seed=seed)

        # Uniform
        error = metric(X_train, y_train, X_test, y_test, random_state=seed)
        rows.append(dict(weight="uniform", error=error, seed=seed))

        # Exact
        error = metric(X_train, y_train, X_test, y_test,
                       sample_weight=r.ratio(X_train).numpy().squeeze(),
                       random_state=seed)
        rows.append(dict(weight="exact", error=error, seed=seed))

        # RuLSIF
        r_rulsif = RuLSIFDensityRatioEstimator(alpha=1e-6)
        r_rulsif.fit(X_test, X_train)
        sample_weight = np.maximum(1e-6, r_rulsif.ratio(X_train))
        error = metric(X_train, y_train, X_test, y_test,
                       sample_weight=sample_weight, random_state=seed)
        rows.append(dict(weight="rulsif", error=error, seed=seed))

        # KLIEP
        r_kliep = KLIEPDensityRatioEstimator(seed=seed)
        r_kliep.fit(X_test, X_train)
        sample_weight = np.maximum(1e-6, r_kliep.ratio(X_train))
        error = metric(X_train, y_train, X_test, y_test,
                       sample_weight=sample_weight, random_state=seed)
        rows.append(dict(weight="kliep", error=error, seed=seed))

        # Logistic Regression (Linear)
        r_linear = MLPDensityRatioEstimator(num_layers=0, num_units=None, seed=seed)
        r_linear.compile(optimizer="adam", metrics=["accuracy"])
        r_linear.fit(X_test, X_train, epochs=250, batch_size=64)
        sample_weight = np.maximum(1e-6, r_linear.ratio(X_train).numpy())
        error = metric(X_train, y_train, X_test, y_test,
                       sample_weight=sample_weight, random_state=seed)
        rows.append(dict(weight="logreg_linear", error=error, seed=seed))

        # Logistic Regression (MLP)
        r_mlp = MLPDensityRatioEstimator(num_layers=2, num_units=32,
                                         activation="tanh", seed=seed)
        r_mlp.compile(optimizer="adam", metrics=["accuracy"])
        r_mlp.fit(X_test, X_train, epochs=250, batch_size=64)
        sample_weight = np.maximum(1e-6, r_mlp.ratio(X_train).numpy())
        error = metric(X_train, y_train, X_test, y_test,
                       sample_weight=sample_weight, random_state=seed)
        rows.append(dict(weight="logreg_deep", error=error, seed=seed))

        # Gaussian Processes
        gpdre = GaussianProcessDensityRatioEstimator(
            input_dim=num_features,
            kernel_cls=kernel_cls,
            vgp_cls=VGP,
            jitter=jitter,
            seed=seed)
        gpdre.compile(optimizer=optimizer)
        gpdre.fit(X_test, X_train)

        for prop_name, prop in props.items():

            r_prop = gpdre.ratio(X_train, convert_to_tensor_fn=prop)
            error = metric(X_train, y_train, X_test, y_test,
                           sample_weight=r_prop.numpy(), random_state=seed)
            rows.append(dict(weight=f"gp_{prop_name}", error=error, seed=seed))

    data = pd.DataFrame(rows)
    data.to_csv(str(summary_path.joinpath(f"{name}.csv")))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
