import tensorflow as tf

from ...benchmarks import SugiyamaKrauledatMuellerDensityRatioMarginals


def get_dataset(num_train, num_test, threshold=0.5, seed=None):

    # Load / create dataset
    def class_posterior(x1, x2):
        return 0.5 * (1 + tf.tanh(x1 - tf.nn.relu(-x2)))

    r = SugiyamaKrauledatMuellerDensityRatioMarginals()
    return r.make_covariate_shift_dataset(
        class_posterior_fn=class_posterior, num_test=num_test,
        num_train=num_train, threshold=threshold, seed=seed)
