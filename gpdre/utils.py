import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import h5py


# shortcuts
tfd = tfp.distributions


def save_hdf5(X_train, y_train, X_test, y_test, filename):

    with h5py.File(filename, 'w') as f:
        f.create_dataset("X_train", data=X_train)
        f.create_dataset("y_train", data=y_train)
        f.create_dataset("X_test", data=X_test)
        f.create_dataset("y_test", data=y_test)


def load_hdf5(filename):

    with h5py.File(filename, 'r') as f:
        X_train = np.array(f.get("X_train"))
        y_train = np.array(f.get("y_train"))
        X_test = np.array(f.get("X_test"))
        y_test = np.array(f.get("y_test"))

    return (X_train, y_train), (X_test, y_test)


def get_steps_per_epoch(num_train, batch_size):

    return num_train // batch_size


def get_kl_weight(num_train, batch_size):

    kl_weight = batch_size / num_train

    return kl_weight


def to_numpy(transformed_variable):

    return tf.convert_to_tensor(transformed_variable).numpy()


def gp_sample_custom(gp, n_samples, seed=None):

    gp_marginal = gp.get_marginal_distribution()

    batch_shape = tf.ones(gp_marginal.batch_shape.rank, dtype=tf.int32)
    event_shape = gp_marginal.event_shape

    sample_shape = tf.concat([[n_samples], batch_shape, event_shape], axis=0)

    base_samples = gp_marginal.distribution.sample(sample_shape, seed=seed)
    gp_samples = gp_marginal.bijector.forward(base_samples)

    return gp_samples


# TODO: deprecate
qs = {
    "same": tfd.Normal(loc=0.0, scale=1.0),
    "scale_lesser": tfd.Normal(loc=0.0, scale=0.6),
    "scale_greater": tfd.Normal(loc=0.0, scale=2.0),
    "loc_different": tfd.Normal(loc=0.5, scale=1.0),
    "additive": tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[0.95, 0.05]),
        components_distribution=tfd.Normal(loc=[0.0, 3.0], scale=[1.0, 1.0])),
    "bimodal": tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[0.4, 0.6]),
        components_distribution=tfd.Normal(loc=[2.0, -3.0], scale=[1.0, 0.5]))
}
