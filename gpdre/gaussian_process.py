import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow

from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Identity, Constant
from tensorflow.keras import optimizers

from .base import DensityRatioBase
from .datasets import make_classification_dataset
from .utils import get_kl_weight

from tqdm import trange


# shortcuts
tfd = tfp.distributions
kernels = tfp.math.psd_kernels


def _add_diagonal_shift(matrix, shift):
    return tf.linalg.set_diag(matrix, tf.linalg.diag_part(matrix) + shift)


class GaussianProcessClassifierGPFlow:

    def __init__(self, input_dim, num_inducing_points,
                 inducing_index_points_initializer,
                 kernel_cls=gpflow.kernels.SquaredExponential,
                 use_ard=True, jitter=1e-6, seed=None, dtype=tf.float64):

        inducing_index_points_initial = (
            inducing_index_points_initializer(shape=(num_inducing_points,
                                                     input_dim), dtype=dtype))

        self.vgp = gpflow.models.SVGP(
            likelihood=gpflow.likelihoods.Bernoulli(invlink=tf.sigmoid),
            inducing_variable=inducing_index_points_initial,
            kernel=kernel_cls())

        self.optimizer = None

        self.jitter = jitter
        self.seed = seed

    def logit_distribution(self, X):

        qf_loc, qf_cov = self.vgp.predict_f(X, full_cov=True)

        qf_scale = tf.linalg.LinearOperatorLowerTriangular(
            tf.linalg.cholesky(_add_diagonal_shift(qf_cov[..., -1],
                                                   self.jitter)),
            is_non_singular=True)

        return tfd.MultivariateNormalLinearOperator(loc=qf_loc[..., -1],
                                                    scale=qf_scale)

    def compile(self, optimizer, num_samples=None):

        self.optimizer = optimizers.get(optimizer)
        self.num_samples = num_samples

    def fit(self, X, y, epochs=1, batch_size=32, shuffle=True, buffer_size=256):

        num_train = len(X)

        self.vgp.num_data = num_train

        @tf.function
        def train_on_batch(X_batch, y_batch):

            variables = self.vgp.trainable_variables

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(variables)
                loss = self.vgp.training_loss((X_batch, y_batch))

            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))

        dataset = tf.data.Dataset.from_tensor_slices((X, y))

        if shuffle:
            dataset = dataset.shuffle(seed=self.seed, buffer_size=buffer_size)

        dataset = dataset.batch(batch_size, drop_remainder=True)

        for epoch in trange(epochs):
            for step, (X_batch, y_batch) in enumerate(dataset):

                train_on_batch(X_batch, y_batch)


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
            initial_value=1.0, bijector=tfp.bijectors.Softplus(),
            dtype=dtype, name="amplitude")
        self.length_scale = tfp.util.TransformedVariable(
            initial_value=1.0, bijector=tfp.bijectors.Softplus(),
            dtype=dtype, name="length_scale", trainable=length_scale_trainable)
        self.scale_diag = tfp.util.TransformedVariable(
            initial_value=np.ones(input_dim), bijector=tfp.bijectors.Softplus(),
            dtype=dtype, name="scale_diag", trainable=scale_diag_trainable)

        base_kernel = kernel_cls(amplitude=self.amplitude,
                                 length_scale=self.length_scale)

        self.kernel = kernels.FeatureScaled(base_kernel,
                                            scale_diag=self.scale_diag)

        self.observation_noise_variance = tfp.util.TransformedVariable(
            initial_value=1e-6, bijector=tfp.bijectors.Softplus(),
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

    def logit_distribution(self, X, jitter=None):

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

    @staticmethod
    def make_likelihood(f, reinterpreted_batch_ndims=1):

        p = tfd.Bernoulli(logits=f)

        return tfd.Independent(
            p, reinterpreted_batch_ndims=reinterpreted_batch_ndims)

    @staticmethod
    def log_likelihood(y, f):

        likelihood = GaussianProcessClassifier.make_likelihood(f)
        return likelihood.log_prob(y)

    def conditional(self, X, sample_shape=(), jitter=None):
        """
        Sample batch of conditional distributions.
        """
        qf = self.logit_distribution(X, jitter=jitter)
        f_samples = qf.sample(sample_shape)

        return self.make_likelihood(f_samples)

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
            qf_batch = self.logit_distribution(X_batch)
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


class GaussianProcessDensityRatioEstimator(DensityRatioBase, GaussianProcessClassifier):

    def fit(self, X_top, X_bot, *args, **kwargs):

        X, y = make_classification_dataset(X_top, X_bot)

        super(GaussianProcessDensityRatioEstimator, self).fit(X, y, *args, **kwargs)

    def logit(self, X, y=None, convert_to_tensor_fn=tfd.Distribution.mean,
              jitter=None):

        qf = self.logit_distribution(X, jitter=jitter)

        return convert_to_tensor_fn(qf)

    def ratio(self, X, y=None, convert_to_tensor_fn=tfd.Distribution.mean,
              jitter=None):

        qr = self.ratio_distribution(X, jitter=jitter)

        return convert_to_tensor_fn(qr)

    def ratio_distribution(self, X, jitter=None, reinterpreted_batch_ndims=1):
        """
        Transformed distribution. Distribution over density ratio at given
        inputs.

        Not to be confused with a "ratio distribution" in the conventional
        sense, which is the distribution of ratio of random variables.
        """
        qf = self.logit_distribution(X, jitter=jitter)

        # TODO(LT): This way of defining the distribution doesn't yield the
        # appropriate samples -- samples from the marginal (instead of the full
        # covariance Gaussian) and is then transformed through the exponential.
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
