import tensorflow_probability as tfp
import numpy as np

from operator import gt

from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from .base import DensityRatioBase, DensityRatioMarginals
from .math import logit

from tqdm import trange

# shortcuts
tfd = tfp.distributions

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

    def logit(self, X, y=None):

        X_tilde = X - np.mean(X, axis=0)
        u = np.dot(X_tilde, self.w)

        return self.scale * u / np.std(u)


class SugiyamaDensityRatio(DensityRatioBase):
    """
    (Sugiyama et al. 2009)
    """
    def __init__(self, feature):

        self.feature = feature

    def logit(self, X, y=None):

        return logit(np.minimum(1.0-EPS, 4.0 * X[..., self.feature]**2))


class SugiyamaKrauledatMuellerDensityRatioMarginals(DensityRatioMarginals):

    def __init__(self):

        train = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=[0.5, 0.5]),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=[[-2.0, 3.0], [2.0, 3.0]], scale_diag=[1.0, 2.0])
        )

        test = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=[0.5, 0.5]),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=[[0.0, -1.0], [4.0, -1.0]])
        )

        return super(SugiyamaKrauledatMuellerDensityRatioMarginals, self) \
            .__init__(top=test, bot=train)


class LabelShift(DensityRatioBase):

    def logit(self, X, y=None):

        target_scaler = StandardScaler()

        return target_scaler.fit_transform(y.reshape(-1, 1)).squeeze(axis=-1)


def get_cortes_splits(metric_callback, train_data, test_data, num_splits=10,
                      max_iter=100, num_seeds=10, tol=0.05, compare_pred=gt,
                      **kwargs):

    X, y = train_data
    X_test, y_test = test_data

    input_dim = X.shape[-1]

    splits = []
    rs = []
    metrics_uniform = []
    metrics_exact = []

    for split in trange(max_iter):

        if len(splits) >= num_splits:

            break

        r = CortesDensityRatio(input_dim=input_dim, seed=split, **kwargs)

        metrics_uniform_seeds = []
        metrics_exact_seeds = []

        for seed in range(num_seeds):

            # TODO(LT): support option to evaluate on (X_val, y_val)
            (X_train, y_train), (X_val, y_val) = r.train_test_split(X, y, seed=seed)

            metrics_uniform_seeds.append(metric_callback(X_train, y_train, X_test, y_test))
            metrics_exact_seeds.append(metric_callback(X_train, y_train, X_test, y_test,
                                                       sample_weight=np.maximum(1e-6, r.ratio(X_train))))

        mean_uniform = np.mean(metrics_uniform_seeds)
        mean_exact = np.mean(metrics_exact_seeds)

        # TODO: support different filter function for classification accuracy
        # (where higher is better)
        if compare_pred(mean_uniform - mean_exact, tol):

            splits.append(split)

            metrics_uniform.append(metrics_uniform_seeds)
            metrics_exact.append(metrics_exact_seeds)

            rs.append(r)

    print("Found {}/{} splits with difference "
          "greater than {} after {}/{} iterations."
          .format(len(splits), num_splits, tol, split + 1, max_iter))

    if len(splits) < num_splits:
        print("Try increasing `max_iter` or "
              "decreasing `num_splits` and/or `tol`.")

    return splits, rs, metrics_uniform, metrics_exact
