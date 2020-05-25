import numpy as np

from sklearn.gaussian_process.kernels import RBF
from cvxopt import matrix, solvers

from ..base import DensityRatioBase
from ..math import logit


class KMMDensityRatioEstimator(DensityRatioBase):

    def __init__(self, kernel_cls=RBF, B=1000.0, eps=None, *args, **kwargs):

        self.B = B
        self.eps = eps

        self.kernel = RBF()
        self.sol = None

    def logit(self, X, y=None):

        return logit(self.ratio(X, y))

    def ratio(self, X, y=None):

        assert self.sol is not None, "must call `fit` first!"
        # TODO: issue warning that `X` is ignored here
        # since we don't support out-of-sample predictions.

        beta = np.asarray(self.sol['x']).squeeze(axis=-1)

        return beta

    def fit(self, X_top, X_bot, *args, **kwargs):

        num_top = X_top.shape[0]
        num_bot = len(X_bot)

        c = num_bot / num_top

        if self.eps is None:
            s = np.sqrt(num_bot)
            eps = (s - 1) / s
        else:
            eps = self.eps

        K = self.kernel(X_bot, X_bot)
        kappa = c * np.sum(self.kernel(X_bot, X_top), axis=1)

        ones = np.atleast_2d(np.ones(num_bot))
        eye = np.eye(num_bot)

        G = np.vstack([ones, -ones, eye, -eye])
        h = np.r_[num_bot * (eps + 1.0),  # sum_i beta_i <= m(eps+1)
                  num_bot * (eps - 1.0),  # sum_i beta_i > -m(eps-1)
                  np.full(num_bot, self.B),  # beta_i <= B
                  np.zeros(num_bot)]  # beta_i > 0

        # set-up quadratic programming problem
        P = matrix(K)
        q = matrix(-kappa)
        G = matrix(G)
        h = matrix(h)

        self.sol = solvers.qp(P, q, G, h)
