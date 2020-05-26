import sys
import click

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from pathlib import Path
from conf import DATASET_PRETTY_NAMES, WEIGHT_PRETTY_NAMES, WEIGHT_ORDER

GOLDEN_RATIO = 0.5 * (1 + np.sqrt(5))


def pt_to_in(x):

    pt_per_in = 72.27
    return x / pt_per_in


def size(width, aspect=GOLDEN_RATIO):

    width_in = pt_to_in(width)
    return (width_in, width_in / aspect)


WIDTH = 8.0
OUTPUT_DIR = "figures/"


def catplot(data, x="error", kind="strip", weight_order=WEIGHT_ORDER,
            xlabel="test nmse", ax=None):

    if ax is None:
        ax.gca()

    palette = "colorblind"

    if kind == "bar":
        sns.barplot(x=x, y="weight", order=weight_order,
                    # hue="weight", hue_order=weight_order,
                    palette=palette, data=data, ax=ax)
    else:
        if kind == "strip":
            sns.stripplot(x=x, y="weight", order=weight_order,
                          # hue="weight", hue_order=weight_order,
                          palette=palette, dodge=True, jitter=False,
                          alpha=0.25, zorder=1, data=data, ax=ax)
            ci = None
            palette = "dark"
        else:
            ci = 95

        sns.pointplot(x=x, y="weight", order=weight_order,
                      # hue="weight", hue_order=weight_order,
                      palette=palette, dodge=0.67, join=False, ci=ci,
                      markers='d', scale=0.75, data=data, ax=ax)

    # # Improve the legend
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[7:], labels[7:],
    #           title=None, handletextpad=0, columnspacing=1,
    #           ncol=4, frameon=True, bbox_to_anchor=(0.0, 1.05),
    #           loc="lower left", fontsize="xx-small")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("method")


@click.command()
@click.argument("name")
@click.argument("result", type=click.File('r'))
@click.option('--context', default="paper")
@click.option('--width', '-w', type=float, default=WIDTH)
@click.option('--aspect', '-a', type=float, default=GOLDEN_RATIO)
@click.option('--extension', '-e', multiple=True, default=["png"])
@click.option("--output-dir", default=OUTPUT_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(name, result, context, width, aspect, extension, output_dir):

    figsize = size(width, aspect)
    suffix = f"{width:.0f}x{width/aspect:.0f}"

    rc = {
        "figure.figsize": figsize,
        "font.serif": ['Times New Roman'],
        "text.usetex": True,
    }

    sns.set(context=context,
            style="whitegrid",
            palette="colorblind",
            font="serif",
            rc=rc)

    output_path = Path(output_dir).joinpath(name)
    output_path.mkdir(parents=True, exist_ok=True)

    baseline = "uniform"

    data = pd.read_csv(result, index_col=0)

    data = data.assign(error=1.0-data["acc"])
    data.drop(columns=["dataset_seed", "acc"], inplace=True)
    # data.drop(columns="dataset_seed", inplace=True)
    # data.replace({"name": DATASET_PRETTY_NAMES}, inplace=True)

    for kind in ["strip", "point", "bar"]:

        fig, ax = plt.subplots()
        sns.despine(fig, ax, bottom=True, left=True)

        catplot(data.replace({"weight": WEIGHT_PRETTY_NAMES}),
                kind=kind, ax=ax)

        for ext in extension:
            fig.savefig(output_path.joinpath(f"abs_{kind}_{suffix}.{ext}"),
                        bbox_inches="tight")

        plt.show()

    data.set_index(["weight", "seed"], inplace=True)
    data_baseline = data.query(f"weight == '{baseline}'") \
                        .reset_index(level="weight", drop=True)

    data_rel = data.divide(data_baseline, axis="index", level="seed") \
                   .rename(columns={"error": "error_relative"})
    data_rel = data_rel.assign(error_relative_change=1.0 - data_rel.error_relative)

    data_new = pd.concat([data, data_rel], axis="columns", join="inner") \
                 .reset_index()
    # .drop(index=baseline, level="weight") \

    for kind in ["strip", "point", "bar"]:

        fig, ax = plt.subplots()
        sns.despine(fig, ax, bottom=True, left=True)

        catplot(data_new.replace({"weight": WEIGHT_PRETTY_NAMES}),
                x="error_relative_change", kind=kind,
                weight_order=WEIGHT_ORDER[1:],
                # xlabel="test nmse (relative improvement)", ax=ax)
                xlabel="test error rate (relative improvement)", ax=ax)

        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

        for ext in extension:
            fig.savefig(output_path.joinpath(f"rel_{kind}_{suffix}.{ext}"),
                        bbox_inches="tight")

        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
