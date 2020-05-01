import numpy as np

from .base import DensityRatioBase
from .math import logit

from sklearn.utils import check_random_state

# TODO: centralize these numerical constants somewhere
EPS = 1e-7


class HuangDensityRatio(DensityRatioBase):
    """
    (Huang et al. 2007)
    """
    # TODO
    pass


class CortesDensityRatio(DensityRatioBase):
    """
    (Cortes et al. 2008)
    """
    def __init__(self, input_dim, low=-1.0, high=1.0, scale=-4.0, seed=None):

        rng = check_random_state(seed)

        self.w = rng.uniform(low=low, high=high, size=input_dim)
        self.scale = scale

    def logit(self, X):

        X_tilde = X - np.mean(X, axis=0)
        u = np.dot(X_tilde, self.w)

        return self.scale * u / np.std(u)


class SugiyamaDensityRatio(DensityRatioBase):
    """
    (Sugiyama et al. 2009)
    """
    def __init__(self, feature):

        self.feature = feature

    def logit(self, X):

        return logit(np.minimum(1.0-EPS, 4.0 * X[..., self.feature]**2))
