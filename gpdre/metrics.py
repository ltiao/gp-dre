import tensorflow.keras.backend as K

from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from tensorflow.keras.metrics import mean_squared_error as K_mean_squared_error


def normalized_mean_squared_error(y_test, y_pred):

    return mean_squared_error(y_test, y_pred) / y_test.var()


# TODO: this is specifically for use as a Keras metric. Need to rename to be
# consistent or move to another namespace.
def nmse(y_test, y_pred):

    return K_mean_squared_error(y_test, y_pred) / K.var(y_test)


# NMSE as defined Sugiyama et al. 2008 for directly comparing density ratios
def normalized_mean_squared_error_sugiyama(w_true, w_pred):

    w_true_norm = normalize(w_true.reshape(-1, 1), norm="l1", axis=0)
    w_pred_norm = normalize(w_pred.reshape(-1, 1), norm="l1", axis=0)

    return mean_squared_error(w_true_norm, w_pred_norm)
