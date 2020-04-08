from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression


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
