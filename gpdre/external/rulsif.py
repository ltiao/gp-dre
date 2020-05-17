import tensorflow as tf

from densratio import densratio
from ..base import DensityRatioBase


class RuLSIFDensityRatioEstimator(DensityRatioBase):

    def __init__(self, alpha=0.):

        self.alpha = alpha
        self.densratio_obj = None

    def fit(self, X_top, X_bot, *args, **kwargs):

        self.densratio_obj = densratio(X_top, X_bot, alpha=self.alpha)

    def logit(self, X, y=None):

        return tf.math.log(self.densratio_obj.compute_density_ratio(X))
