"""Datasets module."""

import numpy as np

from scipy.special import expit
from sklearn.utils import shuffle as _shuffle
from sklearn.model_selection._split import BaseShuffleSplit
from sklearn.utils import check_random_state


class CovariateShiftSplit(BaseShuffleSplit):

    def __init__(self, test_prob, n_splits=10, random_state=None):

        super(CovariateShiftSplit, self).__init__(
            n_splits=n_splits, test_size=None, train_size=None,
            random_state=random_state)
        self.test_prob = test_prob

    def _iter_indices(self, X, y=None, groups=None):

        rng = check_random_state(self.random_state)

        for i in range(self.n_splits):

            p = self.test_prob(X, rng)
            mask_test = rng.binomial(n=1, p=p).astype(bool)
            mask_train = ~mask_test

            ind_train = np.flatnonzero(mask_train)
            ind_test = np.flatnonzero(mask_test)

            yield ind_train, ind_test


def test_prob_cortes(X, rng):

    _, num_features = X.shape

    X_tilde = X - X.mean(axis=1, keepdims=True)

    # TODO: this assumes `X` has been normalized
    w = rng.uniform(low=-1.0, high=1.0, size=num_features)
    u = X_tilde.dot(w)
    logit = 4.0 * u / u.std()

    return expit(logit)


def test_prob_sugiyama(X, rng):

    _, num_features = X.shape

    feature_index = rng.randint(num_features)

    # TODO: this assumes `X` has been normalized
    return np.minimum(1.0, 4.0 * X[..., feature_index]**2)


def make_classification_dataset(X_top, X_bot, shuffle=False, dtype="float64",
                                random_state=None):

    y_top = np.ones(len(X_top))
    y_bot = np.zeros(len(X_bot))

    X = np.vstack([X_top, X_bot]).astype(dtype)
    y = np.hstack([y_top, y_bot])

    if shuffle:
        X, y = _shuffle(X, y, random_state=random_state)

    return X, y
