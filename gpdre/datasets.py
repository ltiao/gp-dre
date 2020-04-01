"""Datasets module."""

import numpy as np

from sklearn.utils import shuffle as _shuffle


def make_classification_dataset(X_top, X_bot, shuffle=False, dtype="float64",
                                random_state=None):

    y_top = np.ones(len(X_top))
    y_bot = np.zeros(len(X_bot))

    X = np.vstack([X_top, X_bot]).astype(dtype)
    y = np.hstack([y_top, y_bot])

    if shuffle:
        X, y = _shuffle(X, y, random_state=random_state)

    return X, y
