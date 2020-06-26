from pathlib import Path


def get_splits(mat, key, squeeze_axis=-1):

    return list(mat.get(key).squeeze(axis=squeeze_axis))


def get_path(name, kind="data", data_home="results/", as_str=True):

    path = Path(data_home).joinpath(f"{name}_KRR_{kind}.mat")

    if as_str:
        return str(path)

    return path
