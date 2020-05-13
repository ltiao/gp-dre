import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import optimizers
from ..utils import get_kl_weight

from tqdm import trange

# shortcuts
tfd = tfp.distributions
kernels = tfp.math.psd_kernels


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
