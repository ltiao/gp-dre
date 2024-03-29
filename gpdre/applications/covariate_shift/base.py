import tensorflow as tf

from abc import ABC, abstractmethod
from tensorflow.keras.regularizers import l1_l2

from ...base import MLPDensityRatioEstimator
from ...math import logit

# from .initializers import KMeans

# kernels = tfp.math.psd_kernels


class BaseCovariateShiftAdapter(ABC):

    @abstractmethod
    def importance_weights(self, X_train, X_test):
        pass


class ExactCovariateShiftAdapter(BaseCovariateShiftAdapter):

    def __init__(self, exact_density_ratio):

        self.exact_density_ratio = exact_density_ratio

    @classmethod
    def from_logit(cls, true_logit):

        return cls(exact_density_ratio=tf.exp(true_logit))

    @classmethod
    def from_prob(cls, true_prob):

        return cls.from_logit(true_logit=logit(true_prob))

    def importance_weights(self, X_train, X_test):

        return self.exact_density_ratio(X_train).numpy()


# class GaussianProcessCovariateShiftAdapter(BaseCovariateShiftAdapter):

#     def __init__(self, num_features, num_inducing_points,
#                  quadrature_size, optimizer, batch_size, epochs,
#                  kernel_cls=kernels.MaternFiveHalves,
#                  use_ard=True, jitter=1e-6, seed=None):

#         self.quadrature_size = quadrature_size
#         self.optimizer = optimizer
#         self.batch_size = batch_size
#         self.epochs = epochs

#         self.dre = GaussianProcessDensityRatioEstimator(
#             input_dim=num_features,
#             num_inducing_points=num_inducing_points,
#             inducing_index_points_initializer=KMeans(X_train, seed=seed),
#             kernel_cls=kernel_cls, use_ard=use_ard, jitter=jitter, seed=seed)
#         self.dre.compile(optimizer=optimizer, quadrature_size=quadrature_size)

#     def importance_weights(self, X_train, X_test):

#         self.dre.fit(X_test, X_train, epochs=num_epochs, batch_size=batch_size,
#                      buffer_size=shuffle_buffer_size)


class MLPCovariateShiftAdapter(BaseCovariateShiftAdapter):

    def __init__(self, num_layers, num_units, activation, l1_factor, l2_factor,
                 optimizer, epochs, batch_size, seed=None):

        self.batch_size = batch_size
        self.epochs = epochs

        self.estimator = MLPDensityRatioEstimator(num_layers=num_layers,
                                                  num_units=num_units,
                                                  activation=activation,
                                                  kernel_regularizer=l1_l2(l1_factor, l2_factor),
                                                  seed=seed)
        self.estimator.compile(optimizer=optimizer, metrics=["accuracy"])

    def importance_weights(self, X_train, X_test):

        self.estimator.fit(X_test, X_train, epochs=self.epochs,
                           batch_size=self.batch_size)

        # TODO: make function for retrieving weights for single input
        return self.estimator(X_train).numpy().squeeze()

    def accuracy(self, X_train, X_test):
        """
        Accuracy in disambiguating between test and training samples.
        """
        loss, accuracy = self.estimator.evaluate(X_test, X_train)
        return accuracy
