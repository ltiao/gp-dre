import tensorflow_probability as tfp

# from .native import GaussianProcessClassifier
from .gpflow import GaussianProcessClassifier
from ..base import DensityRatioBase
from ..datasets import make_classification_dataset

# shortcuts
tfd = tfp.distributions
kernels = tfp.math.psd_kernels


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
        # appropriate samples -- it just samples from the marginal (instead of
        # the full covariance Gaussian) and then transforms through the
        # exponential.
        qr = tfd.LogNormal(loc=qf.mean(), scale=qf.stddev())

        return tfd.Independent(
            qr, reinterpreted_batch_ndims=reinterpreted_batch_ndims)
