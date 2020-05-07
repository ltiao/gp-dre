"""Datasets module."""

import numpy as np

from sklearn.model_selection._split import BaseShuffleSplit
from sklearn.utils import check_random_state, shuffle as _shuffle


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


def make_classification_dataset(X_top, X_bot, shuffle=False, dtype="float64",
                                random_state=None):

    y_top = np.ones(len(X_top))
    y_bot = np.zeros(len(X_bot))

    X = np.vstack([X_top, X_bot]).astype(dtype)
    y = np.hstack([y_top, y_bot])

    if shuffle:
        X, y = _shuffle(X, y, random_state=random_state)

    return X, y


def train_test_split(X, y, prob, seed=None):

    rng = check_random_state(seed)

    mask_test = rng.binomial(n=1, p=prob).astype(bool)
    mask_train = ~mask_test

    return (X[mask_train], y[mask_train]), (X[mask_test], y[mask_test])
