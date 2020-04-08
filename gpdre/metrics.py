from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error


def normalized_mean_squared_error(w_true, w_pred):

    w_true_normed = normalize(w_true.reshape(-1, 1), norm="l1", axis=0)
    w_pred_normed = normalize(w_pred.reshape(-1, 1), norm="l1", axis=0)

    return mean_squared_error(w_true_normed, w_pred_normed)
