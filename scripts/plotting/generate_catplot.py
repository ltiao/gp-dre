import sys
import click

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from conf import WEIGHT_PRETTY_NAMES, WEIGHT_ORDER

GOLDEN_RATIO = 0.5 * (1 + np.sqrt(5))


def pt_to_in(x):

    pt_per_in = 72.27
    return x / pt_per_in


def size(width, aspect=GOLDEN_RATIO):

    width_in = pt_to_in(width)
    return (width_in, width_in / aspect)


WIDTH = 8.0
OUTPUT_DIR = "logs/figures/"


def catplot(data, strip=False, ax=None):

    if ax is None:
        ax.gca()

    if strip:
        sns.stripplot(x="nmse", y="dataset_name",
                      hue="weight", hue_order=WEIGHT_ORDER,
                      palette="colorblind", dodge=True, jitter=False,
                      alpha=0.25, zorder=1, data=data, ax=ax)
        ci = None
    else:
        ci = "sd"

    sns.pointplot(x="nmse", y="dataset_name",
                  hue="weight", hue_order=WEIGHT_ORDER,
                  palette="dark", dodge=0.67, join=False, ci=ci,
                  markers='d', scale=0.75, data=data, ax=ax)

    # Improve the legend
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, labels,
    #           title=None, handletextpad=0, columnspacing=1,
    #           ncol=4, frameon=True, bbox_to_anchor=(0.0, 1.05),
    #           loc="lower left", fontsize="xx-small")

    ax.set_xlabel("test nmse")


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
        "text.usetex": False,
    }

    sns.set(context=context,
            style="whitegrid",
            palette="colorblind",
            font="serif",
            rc=rc)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(result, index_col=0)
    # data = data.assign(error=1.0-data["acc"])
    data.replace({"weight": WEIGHT_PRETTY_NAMES}, inplace=True)

    fig, ax = plt.subplots()
    sns.despine(fig, ax, bottom=True, left=True)

    catplot(data, ax)

    for ext in extension:
        fig.savefig(output_path.joinpath(f"{name}_{suffix}.{ext}"),
                    bbox_inches="tight")

    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
