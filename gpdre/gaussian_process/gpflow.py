import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from gpflow.models import SVGP
from gpflow.kernels import SquaredExponential
from gpflow.likelihoods import Bernoulli

from tensorflow.keras import optimizers

from tqdm import trange

# shortcuts
tfd = tfp.distributions
kernels = tfp.math.psd_kernels


class DummyKernel(kernels.ExponentiatedQuadratic):

    pass


class VariationalGaussianProcessWrapper(tfd.GaussianProcess):

    def __init__(self, vgp, index_points=None, jitter=1e-6,
                 validate_args=False, allow_nan_stats=False, name="VGPWrapper"):

        self._vgp = vgp

        def _mean_fn(index_points):

            qf_loc, qf_var = self._vgp.predict_f(index_points)

            # TODO: make sure squeeze is correct in all cases
            return tf.squeeze(qf_loc, axis=-1)

        super(VariationalGaussianProcessWrapper, self).__init__(
            kernel=DummyKernel(),  # this kernel is not used at all
            index_points=index_points,
            mean_fn=_mean_fn,
            jitter=jitter,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name)

    def _variance(self, index_points=None):
        index_points = self._get_index_points(index_points)
        qf_loc, qf_var = self._vgp.predict_f(index_points)

        # TODO: make sure squeeze is correct in all cases
        return tf.squeeze(qf_var, axis=-1)

    def _compute_covariance(self, index_points):
        qf_loc, qf_cov = self._vgp.predict_f(index_points, full_cov=True)

        # TODO: make sure squeeze is correct in all cases
        return tf.squeeze(qf_cov, axis=0)


class GaussianProcessClassifier:

    def __init__(self, input_dim, num_inducing_points,
                 inducing_index_points_initializer,
                 kernel_cls=SquaredExponential, use_ard=True,
                 vgp_cls=SVGP, whiten=True, jitter=1e-6, seed=None,
                 dtype=tf.float64):

        inducing_index_points_initial = (
            inducing_index_points_initializer(shape=(num_inducing_points,
                                                     input_dim), dtype=dtype))

        # TODO: `use_ard` currently does nothing.
        # TODO: jitter is currently not propagated to GPFlow. Only used
        #   subsequently to compute cholesky to instantiate marginal
        #   distribution.
        # TODO: support mean function
        self._vgp = vgp_cls(
            # data=(observation_index_points, observations),
            kernel=kernel_cls(),
            likelihood=Bernoulli(invlink=tf.sigmoid),
            inducing_variable=inducing_index_points_initial,
            mean_function=None, whiten=whiten, num_data=None)

        # TODO: Support Dense VGP
        # self._vgp = vgp_cls(
        #     data=(observation_index_points, observations),
        #     kernel=kernel,
        #     likelihood=Bernoulli(invlink=tf.sigmoid),  # TODO: this shouldn't be fixed at this level of abstraction
        #     mean_function=None)

        self.optimizer = None

        self.whiten = whiten
        self.jitter = jitter
        self.seed = seed

    def logit_distribution(self, X, jitter=None):

        if jitter is None:
            jitter = self.jitter

        return VariationalGaussianProcessWrapper(self._vgp,
                                                 index_points=X,
                                                 jitter=jitter)

    def compile(self, optimizer, num_samples=None):

        self.optimizer = optimizers.get(optimizer)
        # TODO: issue warning that `num_samples` currently has no effect.
        self.num_samples = num_samples

    def fit(self, X, y, epochs=1, batch_size=32, shuffle=True, buffer_size=256):

        num_train = len(X)
        self._vgp.num_data = num_train

        # TODO(LT): this may not be the most sensible place to do this.
        # Should be done at a higher level of abstraction...
        Y = np.atleast_2d(y).T

        @tf.function
        def train_on_batch(X_batch, y_batch):

            variables = self._vgp.trainable_variables

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(variables)
                loss = self._vgp.training_loss((X_batch, y_batch))

            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))

        dataset = tf.data.Dataset.from_tensor_slices((X, Y))

        if shuffle:
            dataset = dataset.shuffle(seed=self.seed, buffer_size=buffer_size)

        dataset = dataset.batch(batch_size, drop_remainder=True)

        for epoch in trange(epochs):
            for step, (X_batch, y_batch) in enumerate(dataset):

                train_on_batch(X_batch, y_batch)

    # TODO(LT): It might become confusing why these are here for the
    #   GPFlow-based backend engines. Need to streamline class hierachies...
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

    @staticmethod
    def make_likelihood(f, reinterpreted_batch_ndims=1):

        p = tfd.Bernoulli(logits=f)

        return tfd.Independent(
            p, reinterpreted_batch_ndims=reinterpreted_batch_ndims)
