import tensorflow as tf


def logit(p):
    """
    The logit, or log-odds function. Inverse of the logistic sigmoid function.
    """
    return tf.math.log(p) - tf.math.log1p(-p)
