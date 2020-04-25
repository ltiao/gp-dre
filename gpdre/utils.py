import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import pandas as pd
import h5py

from tensorflow.keras.metrics import binary_accuracy
from .datasets import make_classification_dataset \
    as _make_classification_dataset

from pathlib import Path

# shortcuts
tfd = tfp.distributions


def save_hdf5(X_train, y_train, X_test, y_test, filename):

    with h5py.File(filename, 'w') as f:
        f.create_dataset("X_train", data=X_train)
        f.create_dataset("y_train", data=y_train)
        f.create_dataset("X_test", data=X_test)
        f.create_dataset("y_test", data=y_test)


def load_hdf5(filename):

    with h5py.File(filename, 'r') as f:
        X_train = np.array(f.get("X_train"))
        y_train = np.array(f.get("y_train"))
        X_test = np.array(f.get("X_test"))
        y_test = np.array(f.get("y_test"))

    return (X_train, y_train), (X_test, y_test)


class DensityRatio:

    def __init__(self, logit_fn):

        self.logit_fn = logit_fn

    def __call__(self, x):

        return tf.exp(self.logit(x))

    def logit(self, x):
        return self.logit_fn(x)

    def prob(self, x):

        return tf.sigmoid(self.logit(x))


class DensityRatioMarginals(DensityRatio):

    def __init__(self, top, bot, seed=None):

        self.top = top
        self.bot = bot
        self.rng = check_random_state(seed)

    def logit(self, x):

        return self.top.log_prob(x) - self.bot.log_prob(x)

    def train_test_split(self, X, y):

        mask_test = self.rng.binomial(n=1, p=self.prob(X).numpy()).astype(bool)
        mask_train = ~mask_test

        return (X[mask_train], y[mask_train]), (X[mask_test], y[mask_test])


# TODO: make better name
class DensityRatioFoo(DensityRatio):

    def __init__(self, top, bot):

        self.top = top
        self.bot = bot

    def logit(self, x):

        return self.top.log_prob(x) - self.bot.log_prob(x)

    @classmethod
    def from_covariate_shift_example(cls):

        train = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=[0.5, 0.5]),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=[[-2.0, 3.0], [2.0, 3.0]], scale_diag=[1.0, 2.0])
        )

        test = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=[0.5, 0.5]),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=[[0.0, -1.0], [4.0, -1.0]])
        )

        return cls(top=test, bot=train)

    def make_dataset(self, num_samples, rate=0.5, dtype="float64", seed=None):

        num_top = int(num_samples * rate)
        num_bot = num_samples - num_top

        X_top = self.top.sample(sample_shape=(num_top, 1), seed=seed).numpy()
        X_bot = self.bot.sample(sample_shape=(num_bot, 1), seed=seed).numpy()

        return X_top, X_bot

    def make_classification_dataset(self, num_samples, rate=0.5,
                                    dtype="float64", seed=None):

        X_p, X_q = self.make_dataset(num_samples, rate, dtype, seed)
        X, y = _make_classification_dataset(X_p, X_q, dtype=dtype,
                                            random_state=seed)

        return X, y

    def make_covariate_shift_dataset(self, class_posterior_fn, num_test,
                                     num_train, threshold=0.5, seed=None):

        num_samples = num_test + num_train
        rate = num_test / num_samples

        X_test, X_train = self.make_dataset(num_samples, rate=rate, seed=seed)
        # TODO: Temporary fix. Need to address issue in `DistributionPair`.
        X_train = X_train.squeeze()
        X_test = X_test.squeeze()
        y_train = (class_posterior_fn(*X_train.T) > threshold).numpy()
        y_test = (class_posterior_fn(*X_test.T) > threshold).numpy()

        return (X_train, y_train), (X_test, y_test)

    def optimal_accuracy(self, x_test, y_test):

        # Required when some distributions are inherently `float32` such as
        # the `MixtureSameFamily`.
        # TODO: Add flexibility for whether to cast to `float64`.
        y_pred = tf.cast(tf.squeeze(self.prob(x_test)),
                         dtype=tf.float64)

        return binary_accuracy(y_test, y_pred)

    def kl_divergence(self):

        return tfd.kl_divergence(self.p, self.q)


qs = {
    "same": tfd.Normal(loc=0.0, scale=1.0),
    "scale_lesser": tfd.Normal(loc=0.0, scale=0.6),
    "scale_greater": tfd.Normal(loc=0.0, scale=2.0),
    "loc_different": tfd.Normal(loc=0.5, scale=1.0),
    "additive": tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[0.95, 0.05]),
        components_distribution=tfd.Normal(loc=[0.0, 3.0], scale=[1.0, 1.0])),
    "bimodal": tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[0.4, 0.6]),
        components_distribution=tfd.Normal(loc=[2.0, -3.0], scale=[1.0, 0.5]))
}


def get_steps_per_epoch(num_train, batch_size):

    return num_train // batch_size


def get_kl_weight(num_train, batch_size):

    kl_weight = batch_size / num_train

    return kl_weight


def to_numpy(transformed_variable):

    return tf.convert_to_tensor(transformed_variable).numpy()


def gp_sample_custom(gp, n_samples, seed=None):

    gp_marginal = gp.get_marginal_distribution()

    batch_shape = tf.ones(gp_marginal.batch_shape.rank, dtype=tf.int32)
    event_shape = gp_marginal.event_shape

    sample_shape = tf.concat([[n_samples], batch_shape, event_shape], axis=0)

    base_samples = gp_marginal.distribution.sample(sample_shape, seed=seed)
    gp_samples = gp_marginal.bijector.forward(base_samples)

    return gp_samples
