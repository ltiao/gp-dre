import sys
import click

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

GOLDEN_RATIO = 0.5 * (1 + np.sqrt(5))


def pt_to_in(x):

    pt_per_in = 72.27
    return x / pt_per_in


def size(width, aspect=GOLDEN_RATIO):

    width_in = pt_to_in(width)
    return (width_in, width_in / aspect)


WIDTH = 8.0

OUTPUT_DIR = "logs/figures/"

WEIGHT_PRETTY_NAMES = {
    'uniform': "Uniform",
    'exact': "Exact",
    'gp_mean': "GP mean",
    'gp_mode': "GP mode",
    'rulsif': "RuLSIF",
    'kliep': "KLIEP",
    'logreg_linear': "LogReg (linear)",
    'logreg_deep': "LogReg (deep)",
}

WEIGHT_ORDER = [
    'Uniform',
    'Exact',
    'GP mean',
    'GP mode',
    'RuLSIF',
    'KLIEP',
    'LogReg (linear)',
    'LogReg (deep)'
]


def catplot(data, ax=None):

    if ax is None:
        ax.gca()

    # sns.stripplot(x="error", y="name", hue="weight", hue_order=WEIGHT_ORDER,
    #               palette="colorblind", dodge=True, jitter=False,
    #               alpha=0.25, zorder=1, data=data, ax=ax)
    sns.pointplot(x="error", y="name", hue="weight", hue_order=WEIGHT_ORDER,
                  palette="dark", dodge=0.666, join=False,  # ci=None,
                  markers='d', scale=1.0, data=data, ax=ax)

    # Improve the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,
              title=None, handletextpad=0, columnspacing=1,
              ncol=4, frameon=True, bbox_to_anchor=(0.0, 1.05),
              loc="lower left", fontsize="xx-small")

    ax.set_xlabel("test error")


@click.command()
@click.argument("results", type=click.File('r'), nargs=-1)
@click.argument('table', type=click.File('w'))
@click.option('--width', '-w', type=float, default=WIDTH)
@click.option('--aspect', '-a', type=float, default=GOLDEN_RATIO)
@click.option('--extension', '-e', multiple=True, default=["png"])
@click.option("--output-dir", default=OUTPUT_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(results, table, width, aspect, extension, output_dir):

    rc = {
        "figure.figsize": size(width, aspect),
        "font.serif": ['Times New Roman'],
        "text.usetex": True,
    }

    sns.set(context="paper",
            style="whitegrid",
            palette="colorblind",
            font="serif",
            rc=rc)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    names = ["foo", "bar"]
    frames = []

    for name, result in zip(names, results):

        frames.append(pd.read_csv(result, index_col=0).assign(name=name))

    data = pd.concat(frames, axis="index", ignore_index=True, sort=True)
    data = data.assign(error=1.0-data["acc"])
    data.replace({"weight": WEIGHT_PRETTY_NAMES}, inplace=True)

    fig, ax = plt.subplots()
    sns.despine(fig, ax, bottom=True, left=True)

    catplot(data, ax)

    for ext in extension:
        fig.savefig(output_path.joinpath(f"test.{ext}"), bbox_inches="tight")

    plt.show()

    summary = data.groupby(["name", "weight"])["error"] \
                  .describe() \
                  .reset_index() \
                  .pivot(index="weight", columns="name", values=["mean", "std"])

    table.write(summary.to_latex(
        float_format="{:0.3f}".format,
        caption="Synthetic problems. Test error across 10 trials.",
        label="tab:synthetic-results",
        formatters={"std": r"($\pm${:0.3f})".format},
        escape=False)
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
