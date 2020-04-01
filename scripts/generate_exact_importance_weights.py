"""Console script for zalando_classification."""
import h5py
import sys
import click

from etudes.utils import load_hdf5
from etudes.utils import DistributionPair


@click.command()
@click.argument("filename", type=click.Path(dir_okay=False))
@click.argument("dataset_filename", type=click.Path(exists=True, dir_okay=False))
def main(filename, dataset_filename):

    (X_train, y_train), (X_test, y_test) = load_hdf5(dataset_filename)

    pair = DistributionPair.from_covariate_shift_example()
    importance_weights = pair.density_ratio(X_train).numpy()

    with h5py.File(filename, 'w') as f:
        f.create_dataset("importance_weights", data=importance_weights)

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
