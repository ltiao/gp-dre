"""Console script for zalando_classification."""
import tensorflow as tf

import h5py
import sys
import click

from gpdre.utils import DistributionPair, save_hdf5

NUM_TRAIN = 500
NUM_TEST = 500

THRESHOLD = 0.5

SEED = 8888


@click.command()
@click.argument('filename', type=click.Path(dir_okay=False))
@click.option("--importance-weights-filename", type=click.Path(dir_okay=False))
@click.option("--num-train", default=NUM_TRAIN, type=int,
              help="Number of training samples")
@click.option("--num-test", default=NUM_TEST, type=int,
              help="Number of test samples")
@click.option("--threshold", default=THRESHOLD, type=float, help="Threshold")
@click.option("-s", "--seed", default=SEED, type=int, help="Random seed")
def main(filename, importance_weights_filename, num_train, num_test, threshold,
         seed):

    def class_posterior(x1, x2):
        return 0.5 * (1 + tf.tanh(x1 - tf.nn.relu(-x2)))

    pair = DistributionPair.from_covariate_shift_example()
    (X_train, y_train), (X_test, y_test) = pair.make_covariate_shift_dataset(
        class_posterior_fn=class_posterior, num_test=num_test,
        num_train=num_train, threshold=threshold, seed=seed)

    save_hdf5(X_train, y_train, X_test, y_test, filename)

    if importance_weights_filename is not None:

        importance_weights = pair.density_ratio(X_train).numpy()

        with h5py.File(importance_weights_filename, 'w') as f:
            f.create_dataset("importance_weights", data=importance_weights)

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
