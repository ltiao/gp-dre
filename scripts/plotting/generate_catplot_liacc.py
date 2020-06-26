import sys
import click

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from scipy.io import loadmat
from itertools import chain, product
from pathlib import Path

from .conf import DATASET_PRETTY_NAMES, WEIGHT_PRETTY_NAMES, WEIGHT_ORDER
from ..utils import get_path, get_splits

GOLDEN_RATIO = 0.5 * (1 + np.sqrt(5))
WIDTH = 397.48499
OUTPUT_DIR = "figures/"
DODGE = 2/3


def pt_to_in(x):

    pt_per_in = 72.27
    return x / pt_per_in


def size(width, aspect=GOLDEN_RATIO):

    width_in = pt_to_in(width)
    return (width_in, width_in / aspect)


def make_data(mat, key):

    data = pd.DataFrame(data=mat.get(key),
                        columns=mat.get("kw_factors").squeeze(axis=0))

    data.index.name = "split"
    data.columns.name = "kw_factor"

    s = data.stack()
    s.name = "error"

    return s.reset_index()


def generate_key(base="NMSE", method="KLIEP", cv=False, projection=None):

    s = []
    s.append(base)
    s.append(method)

    if cv:
        s.append("CV")

    if projection != "none":
        s.append(projection)

    return '_'.join(s)


@click.command()
# @click.argument("name")
@click.argument("result", type=click.File('r'))
@click.argument("result2", type=click.File('r'))
@click.argument("table", type=click.File('w'))
@click.option('--context', default="paper")
@click.option('--style', default="whitegrid")
@click.option('--width', '-w', type=float, default=WIDTH)
@click.option('--aspect', '-a', type=float, default=GOLDEN_RATIO)
@click.option('--dodge', '-d', type=float, default=DODGE)
@click.option('--extension', '-e', multiple=True, default=["png"])
@click.option("--output-dir", default=OUTPUT_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(result, result2, table, context, style, width, aspect, dodge, extension, output_dir):

    figsize = width_in, height_in = size(width, aspect)
    suffix = f"{width:.0f}x{width/aspect:.0f}"

    rc = {
        "figure.figsize": figsize,
        "font.serif": ['Times New Roman'],
        "text.usetex": True,
    }

    sns.set(context=context,
            style=style,
            palette="colorblind",
            font="serif",
            rc=rc)

    output_path = Path(output_dir).joinpath("regression_liacc_results")
    output_path.mkdir(parents=True, exist_ok=True)

    data_gp = pd.read_csv(result, index_col=0) \
                .query("kernel_name == 'sqr_exp' and projection == 'none'") \
                .rename(columns={"name": "dataset_name", "weight": "method"}) \
                .assign(cv=False)

    data_gp2 = pd.read_csv(result2, index_col=0) \
                 .rename(columns={"name": "dataset_name", "weight": "method"}) \
                 .assign(cv=False)
    # .query("kernel_name == 'sqr_exp' and projection == 'none'") \

    methods = ["KLIEP", "RuLSIF", "cov"]
    projections = ["none", "low", "PCA"]

    frames = []

    for dataset_name in DATASET_PRETTY_NAMES:

        mat = loadmat(get_path(dataset_name, kind="results_CV",
                               data_home="results/20200530/"))

        for method in methods:
            for projection in projections:
                key = generate_key(method=method, projection=projection)
                df = make_data(mat, key=key).assign(dataset_name=dataset_name,
                                                    method=method,
                                                    projection=projection,
                                                    cv=False)

                frames.append(df)

    data_bw_factors = pd.concat(frames, axis="index", sort=True)

    data_bw_factors_display = (
        data_bw_factors.replace({"method": WEIGHT_PRETTY_NAMES,
                                 "dataset_name": DATASET_PRETTY_NAMES})
                       .rename(columns={"kw_factor": "bandwidth factor",
                               "dataset_name": "dataset"}))

    kind = "point"
    # g = sns.catplot(data=data)
    g = sns.catplot(x="dataset", y="error", hue="bandwidth factor",
                    row="method", col="projection",
                    kind=kind, palette="colorblind",
                    dodge=dodge, join=False, scale=0.75, markers='d',
                    data=data_bw_factors_display, sharey="row",
                    height=height_in, aspect=aspect)
    sns.despine(bottom=True, left=True)

    for ext in extension:
        g.savefig(output_path.joinpath(f"bandwidth_factors_{kind}_{suffix}.{ext}"))

    rows = []
    for dataset_name in DATASET_PRETTY_NAMES:

        mat = loadmat(get_path(dataset_name, kind="results_CV",
                               data_home="results/20200530/"))

        for method, projection in product(["KLIEP", "RuLSIF"],
                                          ["none", "PCA", "low"]):

            squeeze_axis = 0 if method == "RuLSIF" else -1

            key = generate_key(method=method, projection=projection, cv=True)
            for split, error in enumerate(mat[key].squeeze(axis=squeeze_axis)):
                row = dict(dataset_name=dataset_name, split=split, cv=True,
                           error=error, method=method, projection=projection)
                rows.append(row)

        squeeze_axis = -1

        for method in ["ones", "ideal"]:

            key = generate_key(method=method, projection="none", cv=False)
            for split, error in enumerate(mat[key].squeeze(axis=squeeze_axis)):
                row = dict(dataset_name=dataset_name, split=split, cv=False,
                           error=error, method=method, projection="none")
                rows.append(row)
                # inject data for projection = "low" as well for visualization
                # row = dict(dataset_name=dataset_name, split=split, cv=False,
                #            error=error, method=method, projection="low")
                # rows.append(row)

    # Results for RuLSIF and KLIEP with `kw_factor` obtained from CV,
    # and results for Uniform and Exact.
    data_cv = pd.DataFrame(rows)

    # Keep only KMM results with the specific `kw_factor`. Omit results for
    # RuLSIF and KLIEP.
    bw_factor = 1.0
    data_bw_factors_fixed = data_bw_factors.query(f"kw_factor == {bw_factor} and method == 'cov'") \
                                           .drop(columns="kw_factor")

    data = pd.concat([data_bw_factors_fixed, data_cv, data_gp, data_gp2],
                     axis="index", join="inner", ignore_index=True) \
             .query("projection != 'PCA'")

    data_display = (
        data.replace({"method": WEIGHT_PRETTY_NAMES,
                      "dataset_name": DATASET_PRETTY_NAMES})
            .rename(columns={"kw_factor": "bandwidth factor",
                             "dataset_name": "dataset"}))

    g = sns.catplot(x="error", y="dataset",
                    hue="method", hue_order=WEIGHT_ORDER[:-2],
                    col="projection", kind=kind, palette="colorblind",
                    dodge=dodge, join=False, scale=0.5, errwidth=1.0, markers='d',
                    data=data_display, sharex="col",
                    height=height_in, aspect=aspect)
    sns.despine(bottom=True, left=True)

    for ext in extension:
        g.savefig(output_path.joinpath(f"abs_{kind}_{suffix}.{ext}"))

    g = sns.catplot(x="dataset", y="error",
                    hue="method", hue_order=WEIGHT_ORDER[:-2],
                    col="projection", kind=kind, palette="colorblind",
                    dodge=dodge, join=False, scale=0.5, errwidth=1.0, markers='d',
                    data=data_display, sharey="row",
                    height=height_in, aspect=aspect)
    g = g.set_xticklabels(rotation=45, horizontalalignment="right")
    sns.despine(bottom=True, left=True)
    # sns.despine(trim=True)

    for ext in extension:
        g.savefig(output_path.joinpath(f"transposed_abs_{kind}_{suffix}.{ext}"))

    summary = data_display.pivot_table(index="dataset",
                                       columns=["projection", "method"],
                                       values="error")
    # aggfunc={"mean": np.mean, "std": np.std})

    table.write(summary.to_latex(float_format="{:0.3f}".format, escape=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
