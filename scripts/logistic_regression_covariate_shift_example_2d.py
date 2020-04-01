"""Console script for zalando_classification."""
import sys
import click

import numpy as np
import pandas as pd
import h5py

from sklearn.linear_model import LogisticRegression
from etudes.utils import load_hdf5
from pathlib import Path

# Sensible defaults
EPOCHS = 1000
OPTIMIZER = "lbfgs"

SUMMARY_DIR = "logs/"

SEED = 0


@click.command()
@click.argument("name")
@click.argument("dataset_filename", type=click.Path(exists=True, dir_okay=False))
@click.option("--sample-weights-filename",
              type=click.Path(exists=True, dir_okay=False),
              help="Name of HDF5 file containing sample weights.")
@click.option("--optimizer", default=OPTIMIZER)
@click.option("-e", "--epochs", default=EPOCHS, type=int,
              help="Number of epochs.")
@click.option("--summary-dir", default=SUMMARY_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Summary directory.")
@click.option("-s", "--seed", default=SEED, type=int, help="Random seed")
def main(name, dataset_filename, sample_weights_filename, optimizer, epochs,
         summary_dir, seed):

    (X_train, y_train), (X_test, y_test) = load_hdf5(dataset_filename)

    sample_weights = None

    if sample_weights_filename is not None:

        # sample_weights = np.ones(len(X_train))
        with h5py.File(sample_weights_filename, 'r') as f:
            sample_weights = np.array(f.get("importance_weights"))

    summary_path = Path(summary_dir).joinpath(name)
    summary_path.mkdir(parents=True, exist_ok=True)

    # Model specification
    model = LogisticRegression(C=1.0, penalty="l2", solver=optimizer,
                               max_iter=epochs, random_state=seed)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Model evaluation
    test_accuracy = model.score(X_test, y_test)

    click.secho(f"[Seed {seed:04d}] test accuracy: {test_accuracy:.3f}",
                fg="green")

    data = pd.Series(dict(optimizer=optimizer, epochs=epochs,
                          test_accuracy=test_accuracy, seed=seed))
    data.to_json(str(summary_path.joinpath(f"{seed:04d}.json")))

    # checkpoint_path = Path(checkpoint_dir).joinpath(name)
    # checkpoint_path.mkdir(parents=True, exist_ok=True)

    # model.save_weights(str(checkpoint_path.joinpath(f"weights.{seed:04d}.h5")))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
