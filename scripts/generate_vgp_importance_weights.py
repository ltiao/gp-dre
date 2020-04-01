"""Console script for zalando_classification."""
import h5py
import sys
import click

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from collections import defaultdict
from tqdm import trange

from tensorflow.keras.layers import InputLayer

from gpdre.gaussian_processes import VariationalGaussianProcessScalar, KernelWrapper
from gpdre.datasets import make_classification_dataset
from gpdre.initializers import KMeans
from gpdre.utils import load_hdf5, get_kl_weight, to_numpy

tfd = tfp.distributions
kernels = tfp.math.psd_kernels

# TODO: add support for option
kernel_cls = kernels.MaternFiveHalves

NUM_INDUCING_POINTS = 50
NUM_QUERY_POINTS = 256

NOISE_VARIANCE = 1e-1
JITTER = 1e-6

QUADRATURE_SIZE = 20
NUM_EPOCHS = 500
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 256

LEARNING_RATE = 1e-3
BETA1 = 0.9
BETA2 = 0.99

SEED = 8888


class GaussianProcessClassifier:

    def __init__(self, input_dim, num_inducing_points,
                 inducing_index_points_initializer, use_ard=True,
                 jitter=1e-6, dtype=tf.float64, seed=None):

        # TODO: should support an optional kernel argument, and only
        # instantiate a new kernel if this argument is not provided.
        # TODO: Add options for initial values of each parameter.
        # TODO: Add option for different bijectors, particular SoftPlus.
        self.amplitude = tfp.util.TransformedVariable(
            initial_value=1.0, bijector=tfp.bijectors.Exp(),
            dtype=dtype, name="amplitude")
        self.length_scale = tfp.util.TransformedVariable(
            initial_value=1.0, bijector=tfp.bijectors.Exp(),
            dtype=dtype, name="length_scale")
        self.scale_diag = tfp.util.TransformedVariable(
            initial_value=np.ones(input_dim), bijector=tfp.bijectors.Exp(),
            dtype=dtype, name="scale_diag")

        self.base_kernel = kernel_cls(amplitude=self.amplitude,
                                      length_scale=self.length_scale)

        if input_dim > 1 and use_ard:
            self.kernel = kernels.FeatureScaled(self.base_kernel,
                                                scale_diag=self.scale_diag)
        else:
            self.kernel = self.base_kernel

        self.observation_noise_variance = tfp.util.TransformedVariable(
            initial_value=1e-3, bijector=tfp.bijectors.Exp(),
            dtype=dtype, name="observation_noise_variance")

        self.inducing_index_points = tf.Variable(
            inducing_index_points_initializer(
                shape=(num_inducing_points, input_dim), dtype=dtype),
            name="inducing_index_points")

        self.variational_inducing_observations_loc = tf.Variable(
            np.zeros(num_inducing_points),
            name="variational_inducing_observations_loc")

        self.variational_inducing_observations_scale = tf.Variable(
            np.eye(num_inducing_points),
            name="variational_inducing_observations_scale"
        )

        self.jitter = jitter
        self.seed = seed

    def __call__(self, X, jitter=None):

        if jitter is None:
            jitter = self.jitter

        return tfd.VariationalGaussianProcess(
            kernel=self.kernel, index_points=X,
            inducing_index_points=self.inducing_index_points,
            variational_inducing_observations_loc=self.variational_loc,
            variational_inducing_observations_scale=self.variational_scale,
            observation_noise_variance=self.observation_noise_variance,
            jitter=jitter
        )

    @staticmethod
    def make_likelihood(f):

        return tfd.Independent(tfd.Bernoulli(logits=f),
                               reinterpreted_batch_ndims=1)

    @staticmethod
    def log_likelihood(y, f):

        likelihood = GaussianProcessClassifier.make_likelihood(f)
        return likelihood.log_prob(y)

    def predictive(self, X, sample_shape=None, jitter=None):
        """
        Sample a batch of likelihood distributions
            p(y|f^(s)), f^(s) ~ q(f) for s in 1...S
        """
        f = self(X, jitter).sample(sample_shape)
        return self.make_likelihood(f)

    def compile(self, optimizer=None):

        if optimizer is None:
            # TODO: Support parameters
            optimizer = tf.keras.optimizers.Adam()

        self.optimizer = optimizer

    def fit(self, X_train, y_train, num_epochs, batch_size=64,
            quadrature_size=20, buffer_size=256):

        num_train = len(X_train)
        kl_weight = get_kl_weight(num_train, batch_size)

        @tf.function
        def elbo(X_batch, y_batch):

            ell = self(X_batch).surrogate_posterior_expected_log_likelihood(
                observations=y_batch,
                log_likelihood_fn=log_likelihood,
                quadrature_size=quadrature_size)

            kl = self(X_batch).surrogate_posterior_kl_divergence_prior()

            return ell - kl_weight * kl

        @tf.function
        def train_step(X_batch, y_batch):

            with tf.GradientTape() as tape:
                nelbo = - elbo(X_batch, y_batch)
                gradients = tape.gradient(nelbo, vgp.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, vgp.trainable_variables))

            return nelbo

        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
                                 .shuffle(seed=self.seed, buffer_size=buffer_size) \
                                 .batch(batch_size, drop_remainder=True)

        # history = defaultdict(list)

        for epoch in range(num_epochs):

            for step, (X_batch, y_batch) in enumerate(dataset):

                train_step(X_batch, y_batch)

    def score(self, X_test, y_test, num_samples=None, jitter=None):
        likelihood = self.sample_likelihood(X_test, num_samples, jitter)
        return likelihood.log_prob(y_test)

    def predict(self, X_test, num_samples=None, jitter=None):
        likelihood = self.sample_likelihood(X_test, num_samples, jitter)
        # TODO: Add support for othe methods, such as `stddev()`.
        return likelihood.mean()


def make_dre(model):

    def dre(X):

        return tfd.Independent(tfd.LogNormal(loc=model(X).mean(),
                                             scale=model(X).stddev()),
                               reinterpreted_batch_ndims=1)

    return dre


def build_model(input_dim, inputs, num_inducing_points, jitter=1e-6):

    return tf.keras.Sequential([
        InputLayer(input_shape=(input_dim,)),
        VariationalGaussianProcessScalar(
            kernel_wrapper=KernelWrapper(input_dim=input_dim,
                                         kernel_cls=kernel_cls,
                                         dtype=tf.float64),
            num_inducing_points=num_inducing_points,
            inducing_index_points_initializer=KMeans(inputs),
            jitter=jitter)
    ])


@click.command()
@click.argument("filename", type=click.Path(dir_okay=False))
@click.argument("dataset_filename", type=click.Path(exists=True, dir_okay=False))
@click.option("--num-inducing-points", default=NUM_INDUCING_POINTS, type=int,
              help="Number of inducing index points")
@click.option("--noise-variance", default=NOISE_VARIANCE, type=int,
              help="Observation noise variance")
@click.option("-e", "--num-epochs", default=NUM_EPOCHS, type=int,
              help="Number of epochs")
@click.option("-b", "--batch-size", default=BATCH_SIZE, type=int,
              help="Batch size")
@click.option("-q", "--quadrature-size", default=QUADRATURE_SIZE, type=int,
              help="Quadrature size")
@click.option("--learning-rate", default=LEARNING_RATE,
              type=float, help="Learning rate")
@click.option("--beta1", default=BETA1,
              type=float, help="Beta 1 optimizer parameter")
@click.option("--beta2", default=BETA2,
              type=float, help="Beta 2 optimizer parameter")
@click.option("--jitter", default=JITTER, type=float, help="Jitter")
@click.option("-s", "--seed", default=SEED, type=int, help="Random seed")
def main(filename, dataset_filename, num_inducing_points, noise_variance,
         num_epochs, batch_size, quadrature_size, learning_rate, beta1, beta2,
         jitter, seed):

    random_state = np.random.RandomState(seed)

    # Don't get confused -- train and test refer to those of the downstream
    # prediction task. Both train and test are used for training of the DRE
    # here.
    (X_train, y_train), (X_test, y_test) = load_hdf5(dataset_filename)
    X, y = make_classification_dataset(X_test, X_train)

    num_train, input_dim = X.shape

    model = build_model(input_dim=input_dim, inputs=X,
                        num_inducing_points=num_inducing_points, jitter=jitter)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         beta_1=beta1, beta_2=beta2)

    @tf.function
    def nelbo(X_batch, y_batch):

        qf = model(X_batch)

        ell = qf.surrogate_posterior_expected_log_likelihood(
            observations=y_batch,
            log_likelihood_fn=log_likelihood,
            quadrature_size=quadrature_size)

        kl = qf.surrogate_posterior_kl_divergence_prior()
        kl_weight = get_kl_weight(num_train, batch_size)

        return - ell + kl_weight * kl

    @tf.function
    def train_step(X_batch, y_batch):

        with tf.GradientTape() as tape:
            loss = nelbo(X_batch, y_batch)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss

    dataset = tf.data.Dataset.from_tensor_slices((X, y)) \
                             .shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER_SIZE) \
                             .batch(batch_size, drop_remainder=True)

    keys = ["inducing_index_points",
            "variational_inducing_observations_loc",
            "variational_inducing_observations_scale",
            "log_observation_noise_variance",
            "log_amplitude", "log_length_scale"]

    history = defaultdict(list)

    with trange(num_epochs, unit="epoch") as range_epochs:

        for epoch in range_epochs:

            for step, (X_batch, y_batch) in enumerate(dataset):

                loss = train_step(X_batch, y_batch)

            range_epochs.set_postfix(loss=to_numpy(loss))

        history["loss"].append(loss.numpy())

        for key, tensor in zip(keys, model.get_weights()):

            history[key].append(tensor)

    dre = make_dre(model)
    importance_weights = dre(X_train).mode().numpy()

    with h5py.File(filename, 'w') as f:
        f.create_dataset("importance_weights", data=importance_weights)

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
