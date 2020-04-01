"""Console script for zalando_classification."""
import sys
import click

import numpy as np
import h5py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.regularizers import l1_l2

from gpdre.datasets import make_classification_dataset
from gpdre.utils import load_hdf5

# Sensible defaults
EPOCHS = 1000
BATCH_SIZE = 64

NUM_LAYERS = 1
NUM_UNITS = 8
ACTIVATION = "tanh"

OPTIMIZER = "adam"

L1_FACTOR = 0.0
L2_FACTOR = 0.0

SEED = 0


def binary_crossentropy_from_logits(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred, from_logits=True)


def build_model(output_dim, num_layers=1, num_units=8, activation="tanh",
                seed=None, *args, **kwargs):
    """
    Instantiate a multi-layer perceptron with rectangular-shaped hidden
    layers.
    """
    model = Sequential()

    for l in range(num_layers):

        model.add(Dense(num_units,
                        activation=activation,
                        kernel_initializer=glorot_uniform(seed=seed),
                        *args, **kwargs))

    model.add(Dense(output_dim, kernel_initializer=glorot_uniform(seed=seed)))

    return model


@click.command()
@click.argument("filename", type=click.Path(dir_okay=False))
@click.argument("dataset_filename", type=click.Path(exists=True, dir_okay=False))
@click.option("--optimizer", default=OPTIMIZER)
@click.option("-l", "--num-layers", default=NUM_LAYERS, type=int,
              help="Number of hidden layers.")
@click.option("-u", "--num-units", default=NUM_UNITS, type=int,
              help="Number of hidden units.")
@click.option("--activation", default=ACTIVATION, type=str)
@click.option("-e", "--epochs", default=EPOCHS, type=int,
              help="Number of epochs.")
@click.option("-b", "--batch-size", default=BATCH_SIZE, type=int,
              help="Batch size.")
@click.option("--l1-factor", default=L1_FACTOR, type=float,
              help="L1 regularization factor.")
@click.option("--l2-factor", default=L2_FACTOR, type=float,
              help="L2 regularization factor.")
@click.option("-s", "--seed", default=SEED, type=int, help="Random seed")
def main(filename, dataset_filename, optimizer, num_layers, num_units,
         activation, epochs, batch_size, l1_factor, l2_factor, seed):

    random_state = np.random.RandomState(seed)

    # Don't get confused -- train and test refer to those of the downstream
    # prediction task. Both train and test are used for training of the DRE
    # here.
    (X_train, y_train), (X_test, y_test) = load_hdf5(dataset_filename)
    X, y = make_classification_dataset(X_test, X_train)

    # Model specification
    model = build_model(output_dim=1, num_layers=num_layers,
                        num_units=num_units, activation=activation,
                        kernel_regularizer=l1_l2(l1_factor, l2_factor),
                        seed=seed)
    model.compile(loss=binary_crossentropy_from_logits, optimizer=optimizer,
                  metrics=["accuracy"])
    hist = model.fit(X, y, batch_size=batch_size, epochs=epochs, shuffle=True)

    importance_weights = np.exp(model.predict(X_train)).squeeze()

    with h5py.File(filename, 'w') as f:
        f.create_dataset("importance_weights", data=importance_weights)

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
