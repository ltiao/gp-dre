"""Console script for zalando_classification."""
import sys
import click

import numpy as np
import h5py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.initializers import glorot_uniform

from gpdre.utils import load_hdf5
from pathlib import Path

# Sensible defaults
EPOCHS = 100
BATCH_SIZE = 64

NUM_LAYERS = 1
NUM_UNITS = 8
ACTIVATION = "tanh"

OPTIMIZER = "adam"

L1_FACTOR = 0.0
L2_FACTOR = 0.0

SEED = 0

CHECKPOINT_DIR = "models/"
SUMMARY_DIR = "logs/"


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

    model.add(Dense(output_dim, kernel_initializer=glorot_uniform(seed=seed),
                    activation="sigmoid"))

    return model


@click.command()
@click.argument("name")
@click.argument("dataset_filename", type=click.Path(exists=True, dir_okay=False))
@click.option("--sample-weights-filename",
              type=click.Path(exists=True, dir_okay=False),
              help="Name of HDF5 file containing sample weights.")
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
@click.option("--checkpoint-dir", default=CHECKPOINT_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Model checkpoint directory.")
@click.option("--summary-dir", default=SUMMARY_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Summary directory.")
@click.option("-s", "--seed", default=SEED, type=int, help="Random seed")
def main(name, dataset_filename, sample_weights_filename,
         optimizer, num_layers, num_units, activation, epochs, batch_size,
         l1_factor, l2_factor, checkpoint_dir, summary_dir, seed):

    (X_train, y_train), (X_test, y_test) = load_hdf5(dataset_filename)

    sample_weights = None

    if sample_weights_filename is not None:

        # sample_weights = np.ones(len(X_train))
        with h5py.File(sample_weights_filename, 'r') as f:
            sample_weights = np.array(f.get("importance_weights"))

    summary_path = Path(summary_dir).joinpath(name)
    summary_path.mkdir(parents=True, exist_ok=True)

    callbacks = []
    callbacks.append(CSVLogger(str(summary_path.joinpath(f"{seed:04d}.csv"))))
    # callbacks.append(TensorBoard(tensorboard_path, profile_batch=0))

    # Model specification
    model = build_model(output_dim=1, num_layers=num_layers,
                        num_units=num_units, activation=activation,
                        kernel_regularizer=l1_l2(l1_factor, l2_factor),
                        seed=seed)
    model.compile(loss="binary_crossentropy", optimizer=optimizer,
                  metrics=["accuracy"])
    # callbacks = build_callbacks(name, seed, summary_dir, checkpoint_dir,
    #                             checkpoint_period)

    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                     validation_data=(X_test, y_test), callbacks=callbacks,
                     shuffle=True, sample_weight=sample_weights)

    # Model evaluation
    loss_test, acc_test = model.evaluate(X_test, y_test)

    click.secho(f"[Seed {seed:04d}] test accuracy: {acc_test:.3f}, "
                f"test loss {loss_test:.3f}", fg='green')

    checkpoint_path = Path(checkpoint_dir).joinpath(name)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    model.save_weights(str(checkpoint_path.joinpath(f"weights.{seed:04d}.h5")))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
