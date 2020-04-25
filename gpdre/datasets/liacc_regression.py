import pandas as pd

from pathlib import Path


def get_column_names(filename, base_path):

    data = pd.read_csv(base_path.joinpath(filename),
                       sep=r"[^a-zA-Z\d]", header=None, index_col=0,
                       engine="python")

    return list(data.index)


# TODO: name these functions
def foo(filename, base_path, target_column, sep=',', column_names=None):

    data = pd.read_csv(base_path.joinpath(filename),
                       sep=sep, header=None, names=column_names)

    return bar(data, target_column)


def bar(data, target_column):

    X = data.drop(target_column, axis="columns").to_numpy()
    y = data[target_column].to_numpy()

    return X, y


def load_bank32nh(data_home="../datasets/"):

    base_path = Path(data_home).joinpath("liacc", "Bank32nh")
    column_names = get_column_names("bank32nh.domain", base_path)

    X_train, y_train = foo(filename="bank32nh.data",
                           base_path=base_path,
                           target_column="rej",
                           sep=r'\s+', column_names=column_names)
    X_test, y_test = foo(filename="bank32nh.test",
                         base_path=base_path,
                         target_column="rej",
                         sep=r'\s+', column_names=column_names)

    return (X_train, y_train), (X_test, y_test)


def load_bank8fm(data_home="../datasets/"):

    base_path = Path(data_home).joinpath("liacc", "Bank8FM")
    column_names = get_column_names("bank8FM.domain", base_path)

    X_train, y_train = foo(filename="bank8FM.data",
                           base_path=base_path,
                           target_column="rej",
                           sep=r'\s+', column_names=column_names)
    X_test, y_test = foo(filename="bank8FM.test",
                         base_path=base_path,
                         target_column="rej",
                         sep=r'\s+', column_names=column_names)

    return (X_train, y_train), (X_test, y_test)


def load_puma8nh(data_home="../datasets/"):

    base_path = Path(data_home).joinpath("liacc", "pumadyn-8nh")
    column_names = get_column_names("puma8NH.domain", base_path)

    X_train, y_train = foo(filename="puma8NH.data",
                           base_path=base_path,
                           target_column="thetadd3",
                           sep=r'\s+', column_names=column_names)
    X_test, y_test = foo(filename="puma8NH.test",
                         base_path=base_path,
                         target_column="thetadd3",
                         sep=r'\s+', column_names=column_names)

    return (X_train, y_train), (X_test, y_test)


def load_cpu_act(data_home="../datasets/"):

    base_path = Path(data_home).joinpath("liacc", "ComputerActivity")
    column_names = get_column_names("cpu_act.domain", base_path)

    X, y = foo(filename="cpu_act.data",
               base_path=base_path,
               target_column="usr",
               column_names=column_names)

    return X, y


def load_cpu_small(data_home="../datasets/"):

    base_path = Path(data_home).joinpath("liacc", "ComputerActivity")
    column_names = get_column_names("cpu_small.domain", base_path)

    X, y = foo(filename="cpu_small.data",
               base_path=base_path,
               target_column="usr",
               column_names=column_names)

    return X, y


def load_kin8nm(data_home="../datasets/"):

    base_path = Path(data_home).joinpath("liacc", "Kinematics")
    column_names = get_column_names("kin8nm.domain", base_path)

    X, y = foo(filename="kin8nm.data",
               base_path=base_path,
               target_column='y',
               column_names=column_names)

    return X, y


def load_ailerons(data_home="../datasets/"):

    base_path = Path(data_home).joinpath("liacc", "Ailerons")
    column_names = get_column_names("ailerons.domain", base_path)

    X_train, y_train = foo(filename="ailerons.data",
                           base_path=base_path,
                           target_column="goal",
                           column_names=column_names)
    X_test, y_test = foo(filename="ailerons.test",
                         base_path=base_path,
                         target_column="goal",
                         column_names=column_names)

    return (X_train, y_train), (X_test, y_test)


def load_elevators(data_home="../datasets/"):

    base_path = Path(data_home).joinpath("liacc", "Elevators")
    column_names = get_column_names("elevators.domain", base_path)

    X_train, y_train = foo(filename="elevators.data",
                           base_path=base_path,
                           target_column="Goal",
                           column_names=column_names)
    X_test, y_test = foo(filename="elevators.test",
                         base_path=base_path,
                         target_column="Goal",
                         column_names=column_names)

    return (X_train, y_train), (X_test, y_test)


DATASET_LOADER = {
    "ailerons": load_ailerons,
    "bank32nh": load_bank32nh,
    "bank8fm": load_bank8fm,
    "cpu_act": load_cpu_act,
    "cpu_small": load_cpu_small,
    "elevators": load_elevators,
    "kin8nm": load_kin8nm,
    "puma8nh": load_puma8nh
}


def load_dataset(name, *args, **kwargs):

    return DATASET_LOADER.get(name)(*args, **kwargs)
