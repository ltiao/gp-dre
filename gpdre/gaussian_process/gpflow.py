import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from gpflow.models import VGP, VGPOpperArchambeau, SVGP
from gpflow.kernels import Stationary, SquaredExponential
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

    def __init__(self, input_dim, kernel_cls=SquaredExponential, use_ard=True,
                 vgp_cls=SVGP, num_inducing_points=None,
                 inducing_index_points_initializer=None, whiten=True,
                 jitter=1e-6, seed=None, dtype=tf.float64):

        # TODO: We only need this check to avoid passing unexpected
        # `lengthscales` kwarg. The correct thing to do is add
        # logic to not pass this kwarg for non-stationary kernels...
        assert issubclass(kernel_cls, Stationary), \
            "Currently only support stationary kernels."

        if input_dim > 1 and use_ard:
            length_scales = np.ones(input_dim)
        else:
            length_scales = 1.0

        self.kernel = kernel_cls(lengthscales=length_scales)
        self.likelihood = Bernoulli(invlink=tf.sigmoid)

        if inducing_index_points_initializer is not None:
            assert num_inducing_points is not None, \
                "Must specify `input_dim` and `num_inducing_points`."
            self.inducing_index_points_initial = (
                inducing_index_points_initializer(
                    shape=(num_inducing_points, input_dim), dtype=dtype))

        self.whiten = whiten
        self.vgp_cls = vgp_cls

        self._vgp = None
        self.optimizer = None

        self.whiten = whiten
        self.jitter = jitter
        self.seed = seed

    def logit_distribution(self, X, jitter=None):

        assert self._vgp is not None, "Model not yet instantiated! " \
            "Call `fit` first."

        if jitter is None:
            jitter = self.jitter

        return VariationalGaussianProcessWrapper(self._vgp,
                                                 index_points=X,
                                                 jitter=jitter)

    def compile(self, optimizer, num_samples=None):

        if isinstance(optimizer, str):
            self.optimizer = optimizers.get(optimizer)
        else:
            self.optimizer = optimizer

        # TODO: issue warning that `num_samples` currently has no effect.
        self.num_samples = num_samples

    def fit(self, X, y, epochs=1, batch_size=32, shuffle=True, buffer_size=256):

        assert self.optimizer is not None, "optimizer not specified! " \
            "Call `compile` first."

        # TODO(LT): this may not be the most sensible place to do this.
        # Should be done at a higher level of abstraction...
        Y = np.atleast_2d(y).T
        num_train = len(X)

        # TODO: jitter is currently not propagated to GPFlow. Only used
        #   subsequently to compute cholesky to instantiate marginal
        #   distribution.
        # TODO: support mean function
        if self.vgp_cls in [VGP, VGPOpperArchambeau]:

            self._vgp = self.vgp_cls(
                data=(X, Y),
                kernel=self.kernel,
                likelihood=self.likelihood,
                mean_function=None)
            self.optimizer.minimize(self._vgp.training_loss,
                                    variables=self._vgp.trainable_variables)

        elif self.vgp_cls is SVGP:

            self._vgp = self.vgp_cls(
                kernel=self.kernel,
                likelihood=self.likelihood,
                inducing_variable=self.inducing_index_points_initial,
                mean_function=None, whiten=self.whiten, num_data=num_train)

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
