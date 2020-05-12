"""Main module."""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.backend as K

from abc import ABC, abstractmethod

from tensorflow.keras.metrics import binary_accuracy
from sklearn.utils import check_random_state

from .models import DenseSequential
from .losses import binary_crossentropy_from_logits
from .datasets import make_classification_dataset
from .datasets.base import train_test_split

# shortcuts
tfd = tfp.distributions


class DensityRatioBase(ABC):

    def __call__(self, X, y=None):

        return self.ratio(X, y)

    @abstractmethod
    def logit(self, X, y=None):
        pass

    def ratio(self, X, y=None):

        return tf.exp(self.logit(X, y))

    def prob(self, X, y=None):
        """
        Probability of sample being from P_{top}(x) vs. P_{bot}(x).
        """
        return tf.sigmoid(self.logit(X, y))

    def train_test_split(self, X, y, seed=None):

        # TODO: clean up API to support both pure NumPy and TensorFlow 2.0
        # eager computations.
        return train_test_split(X, y, prob=self.prob(X, y).numpy(), seed=seed)


class DensityRatio(DensityRatioBase):

    def __init__(self, logit_fn):

        self.logit_fn = logit_fn

    def logit(self, X, y=None):
        return self.logit_fn(X)


class DensityRatioMarginals(DensityRatioBase):

    def __init__(self, top, bot):

        self.top = top
        self.bot = bot

    def logit(self, X, y=None):

        return self.top.log_prob(X) - self.bot.log_prob(X)

    def make_dataset(self, num_samples, rate=0.5, dtype="float64", seed=None):

        num_top = int(num_samples * rate)
        num_bot = num_samples - num_top

        X_top = self.top.sample(sample_shape=(num_top, 1), seed=seed).numpy()
        X_bot = self.bot.sample(sample_shape=(num_bot, 1), seed=seed).numpy()

        return X_top, X_bot

    def make_classification_dataset(self, num_samples, rate=0.5,
                                    dtype="float64", seed=None):

        X_p, X_q = self.make_dataset(num_samples, rate, dtype, seed)
        X, y = make_classification_dataset(X_p, X_q, dtype=dtype,
                                           random_state=seed)

        return X, y

    def kl_divergence(self):

        return tfd.kl_divergence(self.top, self.bot)

    # TODO(LT): deprecate
    def optimal_accuracy(self, X_test, y_test):

        # Required when some distributions are inherently `float32` such as
        # the `MixtureSameFamily`.
        # TODO: Add flexibility for whether to cast to `float64`.
        y_pred = tf.cast(tf.squeeze(self.prob(X_test)),
                         dtype=tf.float64)

        return binary_accuracy(y_test, y_pred)

    def make_regression_dataset(self, num_test, num_train, latent_fn,
                                noise_scale=1.0, squeeze=True, seed=None):

        rng = check_random_state(seed)

        # TODO(LT): this may not guarantee that `len(X_test) == num_test` and
        # similarly go
        num_samples = num_test + num_train
        rate = num_test / num_samples

        X_test, X_train = self.make_dataset(num_samples, rate=rate, seed=seed)

        # TODO(LT): broadcast to shape of `latent_fn(X)` properly.
        # Currently assumes shape is always `(*, 1)`.
        eps_train = noise_scale * rng.randn(num_train, 1)
        eps_test = noise_scale * rng.randn(num_test, 1)

        Y_train = latent_fn(X_train) + eps_train
        Y_test = latent_fn(X_test) + eps_test

        if squeeze:
            Y_train = np.squeeze(Y_train)
            Y_test = np.squeeze(Y_test)

        return (X_train, Y_train), (X_test, Y_test)

    # TODO(LT): deprecate
    def make_covariate_shift_dataset(self, class_posterior_fn, num_test,
                                     num_train, threshold=0.5, seed=None):

        num_samples = num_test + num_train
        rate = num_test / num_samples

        X_test, X_train = self.make_dataset(num_samples, rate=rate, seed=seed)
        # TODO(LT): Temporary fix. Need to address issue in `DistributionPair`
        X_train = X_train.squeeze()
        X_test = X_test.squeeze()

        # TODO(LT): this should be done by sampling from a Bernoulli instead...
        y_train = (class_posterior_fn(*X_train.T) > threshold).numpy()
        y_test = (class_posterior_fn(*X_test.T) > threshold).numpy()

        return (X_train, y_train), (X_test, y_test)


class MLPDensityRatioEstimator(DensityRatioBase):

    def __init__(self, num_layers=2, num_units=32, activation="tanh",
                 seed=None, *args, **kwargs):

        self.model = DenseSequential(1, num_layers, num_units,
                                     layer_kws=dict(activation=activation))

    def logit(self, X):

        # TODO: time will tell whether squeezing the final axis
        # makes things easier.
        return K.squeeze(self.model(X), axis=-1)

    def compile(self, optimizer, metrics=["accuracy"], *args, **kwargs):

        self.model.compile(optimizer=optimizer,
                           loss=binary_crossentropy_from_logits,
                           metrics=metrics, *args, **kwargs)

    def fit(self, X_top, X_bot, *args, **kwargs):

        X, y = make_classification_dataset(X_top, X_bot)
        return self.model.fit(X, y, *args, **kwargs)

    def evaluate(self, X_top, X_bot, *args, **kwargs):

        X, y = make_classification_dataset(X_top, X_bot)
        return self.model.evaluate(X, y, *args, **kwargs)
