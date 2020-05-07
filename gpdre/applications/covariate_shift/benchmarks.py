from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

from ...metrics import normalized_mean_squared_error


def regression_metric(X_train, y_train, X_test, y_test, sample_weight=None):

    feature_scaler = StandardScaler()

    Z_train = feature_scaler.fit_transform(X_train)
    Z_test = feature_scaler.transform(X_test)

    input_dim = X_train.shape[-1]
    gamma = 1.0 / input_dim  # gamma = 1/D <=> sigma = sqrt(D/2)

    model = KernelRidge(kernel="rbf", gamma=gamma)
    model.fit(Z_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(Z_test)

    return normalized_mean_squared_error(y_test, y_pred)


class CovariateShiftBenchmark(ABC):

    @abstractmethod
    def test_metric(self, train_data, test_data, importance_weights=None):
        pass


class Classification2DCovariateShiftBenchmark(CovariateShiftBenchmark):

    def __init__(self, optimizer="lbfgs", epochs=500, penalty="l2", seed=None):

        self.model = LogisticRegression(C=1.0, penalty=penalty,
                                        solver=optimizer,
                                        max_iter=epochs,
                                        random_state=seed)

    def test_metric(self, train_data, test_data, importance_weights=None):

        X_train, y_train = train_data
        X_test, y_test = test_data

        self.model.fit(X_train, y_train, sample_weight=importance_weights)

        return self.model.score(X_test, y_test)
