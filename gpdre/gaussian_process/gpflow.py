import tensorflow as tf
import tensorflow_probability as tfp

from gpflow.models import VGP, SVGP
from gpflow.likelihoods import Bernoulli

# shortcuts
tfd = tfp.distributions
kernels = tfp.math.psd_kernels


class DummyKernel(kernels.ExponentiatedQuadratic):

    pass


class VGPWrapper(tfd.GaussianProcess):

    def __init__(self, kernel, index_points=None,
                 observation_index_points=None, observations=None,
                 mean_fn=None, jitter=1e-6, vgp_cls=VGP, validate_args=False,
                 allow_nan_stats=False, name="VGPWrapper"):

        # TODO: jitter is currently not propagated to GPFlow. Only used
        #   subsequently to compute cholesky to instantiate marginal
        #   distribution.
        # TODO: support mean function
        self._vgp = vgp_cls(
            data=(observation_index_points, observations),
            kernel=kernel,
            likelihood=Bernoulli(invlink=tf.sigmoid),
            mean_function=None)

        def _mean_fn(index_points):

            qf_loc, qf_var = self._vgp.predict_f(index_points)

            # TODO: make sure squeeze is correct in all cases
            return tf.squeeze(qf_loc, axis=-1)

        super(VGPWrapper, self).__init__(
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

    @property
    def variational_loss(self):
        return self._vgp.training_loss


class SVGPWrapper(tfd.GaussianProcess):

    def __init__(self, kernel, inducing_index_points_initial,
                 index_points=None, mean_fn=None, jitter=1e-6, whiten=True,
                 num_data=None, validate_args=False, allow_nan_stats=False,
                 name="SVGPWrapper"):

        # TODO: jitter is currently not propagated to GPFlow. Only used
        #   subsequently to compute cholesky to instantiate marginal
        #   distribution.
        # TODO: support mean function
        # TODO: support `q_diag`
        self._vgp = SVGP(
            kernel=kernel,
            likelihood=Bernoulli(invlink=tf.sigmoid),
            inducing_variable=inducing_index_points_initial,
            mean_function=None, whiten=whiten, num_data=num_data)

        def _mean_fn(index_points):

            qf_loc, qf_var = self._vgp.predict_f(index_points)

            # TODO: make sure squeeze is correct in all cases
            return tf.squeeze(qf_loc, axis=-1)

        super(SVGPWrapper, self).__init__(
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


# class GaussianProcessClassifierGPFlow:

#     def __init__(self, input_dim, num_inducing_points,
#                  inducing_index_points_initializer,
#                  kernel_cls=SquaredExponential,
#                  use_ard=True, jitter=1e-6, seed=None, dtype=tf.float64):

#         inducing_index_points_initial = (
#             inducing_index_points_initializer(shape=(num_inducing_points,
#                                                      input_dim), dtype=dtype))

#         self._vgp = SVGP(
#             likelihood=Bernoulli(invlink=tf.sigmoid),
#             inducing_variable=inducing_index_points_initial,
#             kernel=kernel_cls())

#         self.optimizer = None

#         self.jitter = jitter
#         self.seed = seed

#     def logit_distribution(self, X):

#         qf_loc, qf_cov = self.vgp.predict_f(X, full_cov=True)

#         qf_scale = tf.linalg.LinearOperatorLowerTriangular(
#             tf.linalg.cholesky(_add_diagonal_shift(qf_cov[..., -1],
#                                                    self.jitter)),
#             is_non_singular=True)

#         return tfd.MultivariateNormalLinearOperator(loc=qf_loc[..., -1],
#                                                     scale=qf_scale)

#     def compile(self, optimizer, num_samples=None):

#         self.optimizer = optimizers.get(optimizer)
#         self.num_samples = num_samples

#     def fit(self, X, y, epochs=1, batch_size=32, shuffle=True, buffer_size=256):

#         num_train = len(X)

#         self.vgp.num_data = num_train

#         @tf.function
#         def train_on_batch(X_batch, y_batch):

#             variables = self.vgp.trainable_variables

#             with tf.GradientTape(watch_accessed_variables=False) as tape:
#                 tape.watch(variables)
#                 loss = self.vgp.training_loss((X_batch, y_batch))

#             gradients = tape.gradient(loss, variables)
#             self.optimizer.apply_gradients(zip(gradients, variables))

#         dataset = tf.data.Dataset.from_tensor_slices((X, y))

#         if shuffle:
#             dataset = dataset.shuffle(seed=self.seed, buffer_size=buffer_size)

#         dataset = dataset.batch(batch_size, drop_remainder=True)

#         for epoch in trange(epochs):
#             for step, (X_batch, y_batch) in enumerate(dataset):

#                 train_on_batch(X_batch, y_batch)
