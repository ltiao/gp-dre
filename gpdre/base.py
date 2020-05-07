"""Main module."""
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Identity, Constant
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras import optimizers

from abc import ABC, abstractmethod

from .models import DenseSequential
from .losses import binary_crossentropy_from_logits
from .datasets import make_classification_dataset
from .datasets.base import train_test_split
from .utils import get_kl_weight

from tqdm import trange

# shortcuts
tfd = tfp.distributions
kernels = tfp.math.psd_kernels


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


class GaussianProcessClassifier:

    def __init__(self, input_dim, num_inducing_points,
                 inducing_index_points_initializer,
                 kernel_cls=kernels.ExponentiatedQuadratic,
                 use_ard=True, jitter=1e-6, seed=None, dtype=tf.float64):

        # TODO: should support an optional kernel argument, and only
        # instantiate a new kernel if this argument is not provided.
        # TODO: Add options for initial values of each parameter.
        # TODO: Add option for different bijectors, particular SoftPlus.
        length_scale_trainable = True
        scale_diag_trainable = False

        if input_dim > 1 and use_ard:

            length_scale_trainable = False
            scale_diag_trainable = True

        self.amplitude = tfp.util.TransformedVariable(
            initial_value=1.0, bijector=tfp.bijectors.Exp(),
            dtype=dtype, name="amplitude")
        self.length_scale = tfp.util.TransformedVariable(
            initial_value=1.0, bijector=tfp.bijectors.Exp(),
            dtype=dtype, name="length_scale", trainable=length_scale_trainable)
        self.scale_diag = tfp.util.TransformedVariable(
            initial_value=np.ones(input_dim), bijector=tfp.bijectors.Exp(),
            dtype=dtype, name="scale_diag", trainable=scale_diag_trainable)

        base_kernel = kernel_cls(amplitude=self.amplitude,
                                 length_scale=self.length_scale)

        self.kernel = kernels.FeatureScaled(base_kernel,
                                            scale_diag=self.scale_diag)

        self.observation_noise_variance = tfp.util.TransformedVariable(
            initial_value=1e-6, bijector=tfp.bijectors.Exp(),
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

        self.optimizer = None

        self.jitter = jitter
        self.seed = seed

    def logit(self, X, jitter=None):

        if jitter is None:
            jitter = self.jitter

        return tfd.VariationalGaussianProcess(
            kernel=self.kernel, index_points=X,
            inducing_index_points=self.inducing_index_points,
            variational_inducing_observations_loc=(
                self.variational_inducing_observations_loc),
            variational_inducing_observations_scale=(
                self.variational_inducing_observations_scale),
            observation_noise_variance=self.observation_noise_variance,
            jitter=jitter)

    def compile(self, optimizer, quadrature_size=20, num_samples=None):

        if num_samples is not None:

            raise NotImplementedError("Monte Carlo estimation of ELL not yet "
                                      "supported!")

        self.optimizer = optimizers.get(optimizer)
        self.quadrature_size = quadrature_size

    def fit(self, X, y, epochs=1, batch_size=32, shuffle=True, buffer_size=256):

        # TODO: check if instance has already called `compile`.
        num_train = len(X)
        kl_weight = get_kl_weight(num_train, batch_size)

        @tf.function
        def elbo(y_batch, qf_batch):

            # TODO: Add support for sampling in addition to quadrature.
            ell = qf_batch.surrogate_posterior_expected_log_likelihood(
                observations=y_batch,
                log_likelihood_fn=GaussianProcessClassifier.log_likelihood,
                quadrature_size=self.quadrature_size)
            kl = qf_batch.surrogate_posterior_kl_divergence_prior()

            return ell - kl_weight * kl

        @tf.function
        def train_on_batch(X_batch, y_batch):

            # TODO: does this need to be in the GradientTape context manager?
            #   Doesn't seem to, but would bea prime suspect if anything
            #   behaves funky down the line...
            qf_batch = self.logit(X_batch)
            variables = qf_batch.trainable_variables

            with tf.GradientTape() as tape:
                loss = - elbo(y_batch, qf_batch)
                gradients = tape.gradient(loss, variables)
                self.optimizer.apply_gradients(zip(gradients, variables))

        dataset = tf.data.Dataset.from_tensor_slices((X, y))

        if shuffle:
            dataset = dataset.shuffle(seed=self.seed, buffer_size=buffer_size)

        dataset = dataset.batch(batch_size, drop_remainder=True)

        for epoch in trange(epochs):
            for step, (X_batch, y_batch) in enumerate(dataset):

                train_on_batch(X_batch, y_batch)

    @staticmethod
    def make_likelihood(f, reinterpreted_batch_ndims=1):

        p = tfd.Bernoulli(logits=f)

        return tfd.Independent(
            p, reinterpreted_batch_ndims=reinterpreted_batch_ndims)

    @staticmethod
    def log_likelihood(y, f):

        likelihood = GaussianProcessClassifier.make_likelihood(f)
        return likelihood.log_prob(y)

    def predictive_sample(self, X, sample_shape=(), jitter=None):
        """
        Sample batch of predictive distributions.
        """
        qf = self.logit(X, jitter=jitter)
        f_samples = qf.sample(sample_shape)

        return self.make_likelihood(f_samples)


class GaussianProcessDensityRatioEstimator(GaussianProcessClassifier):

    def fit(self, X_top, X_bot, epochs=1, batch_size=32, shuffle=True,
            buffer_size=256):

        X, y = make_classification_dataset(X_top, X_bot)

        super(GaussianProcessDensityRatioEstimator, self).fit(
            X, y, epochs=epochs, batch_size=batch_size, shuffle=shuffle,
            buffer_size=buffer_size)

    def __call__(self, X, jitter=None, reinterpreted_batch_ndims=1):
        """
        Transformed distribution. Predictive is probably a misleading name.
        """
        qf = self.logit(X, jitter=jitter)
        qr = tfd.LogNormal(loc=qf.mean(), scale=qf.stddev())

        return tfd.Independent(
            qr, reinterpreted_batch_ndims=reinterpreted_batch_ndims)


# Legacy code below this point
class KernelWrapper(Layer):

    # TODO: Support automatic relevance determination
    def __init__(self, input_dim=1, kernel_cls=kernels.ExponentiatedQuadratic,
                 dtype=None, **kwargs):

        super(KernelWrapper, self).__init__(dtype=dtype, **kwargs)

        self.kernel_cls = kernel_cls

        self.log_amplitude = self.add_weight(
            name="log_amplitude",
            initializer="zeros", dtype=dtype)

        self.log_length_scale = self.add_weight(
            name="log_length_scale",
            initializer="zeros", dtype=dtype)

        self.log_scale_diag = self.add_weight(
            name="log_scale_diag", shape=(input_dim,),
            initializer="zeros", dtype=dtype)

    def call(self, x):
        # Never called -- this is just a layer so it can hold variables
        # in a way Keras understands.
        return x

    @property
    def kernel(self):

        base_kernel = self.kernel_cls(
            amplitude=tf.exp(self.log_amplitude),
            length_scale=tf.exp(self.log_length_scale))

        return kernels.FeatureScaled(base_kernel,
                                     scale_diag=tf.exp(self.log_scale_diag))


class VariationalGaussianProcessScalar(tfp.layers.DistributionLambda):

    def __init__(self, kernel_wrapper, num_inducing_points,
                 inducing_index_points_initializer, mean_fn=None, jitter=1e-6,
                 convert_to_tensor_fn=tfd.Distribution.sample, **kwargs):

        def make_distribution(x):

            return VariationalGaussianProcessScalar.new(
                x, kernel_wrapper=self.kernel_wrapper,
                inducing_index_points=self.inducing_index_points,
                variational_inducing_observations_loc=(
                    self.variational_inducing_observations_loc),
                variational_inducing_observations_scale=(
                    self.variational_inducing_observations_scale),
                mean_fn=self.mean_fn,
                observation_noise_variance=tf.exp(
                    self.log_observation_noise_variance),
                jitter=self.jitter)

        super(VariationalGaussianProcessScalar, self).__init__(
            make_distribution_fn=make_distribution,
            convert_to_tensor_fn=convert_to_tensor_fn,
            dtype=kernel_wrapper.dtype)

        self.kernel_wrapper = kernel_wrapper
        self.inducing_index_points_initializer = inducing_index_points_initializer
        self.num_inducing_points = num_inducing_points
        self.mean_fn = mean_fn
        self.jitter = jitter

        self._dtype = self.kernel_wrapper.dtype

    def build(self, input_shape):

        input_dim = input_shape[-1]

        # TODO: Fix initialization!
        self.inducing_index_points = self.add_weight(
            name="inducing_index_points",
            shape=(self.num_inducing_points, input_dim),
            initializer=self.inducing_index_points_initializer,
            dtype=self.dtype)

        self.variational_inducing_observations_loc = self.add_weight(
            name="variational_inducing_observations_loc",
            shape=(self.num_inducing_points,),
            initializer="zeros", dtype=self.dtype)

        self.variational_inducing_observations_scale = self.add_weight(
            name="variational_inducing_observations_scale",
            shape=(self.num_inducing_points, self.num_inducing_points),
            initializer=Identity(gain=1.0), dtype=self.dtype)

        self.log_observation_noise_variance = self.add_weight(
            name="log_observation_noise_variance",
            initializer=Constant(-5.0), dtype=self.dtype)

    @staticmethod
    def new(x, kernel_wrapper, inducing_index_points, mean_fn,
            variational_inducing_observations_loc,
            variational_inducing_observations_scale,
            observation_noise_variance, jitter, name=None):

        # ind = tfd.Independent(base, reinterpreted_batch_ndims=1)
        # bijector = tfp.bijectors.Transpose(rightmost_transposed_ndims=2)
        # d = tfd.TransformedDistribution(ind, bijector=bijector)

        return tfd.VariationalGaussianProcess(
            kernel=kernel_wrapper.kernel, index_points=x,
            inducing_index_points=inducing_index_points,
            variational_inducing_observations_loc=(
                variational_inducing_observations_loc),
            variational_inducing_observations_scale=(
                variational_inducing_observations_scale),
            mean_fn=mean_fn,
            observation_noise_variance=observation_noise_variance,
            jitter=jitter)
