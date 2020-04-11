import tensorflow as tf


def logit(p):
    """
    The logit, or log-odds function. Inverse of the logistic sigmoid function.
    """
    return tf.log(p) - tf.log1p(-p)
