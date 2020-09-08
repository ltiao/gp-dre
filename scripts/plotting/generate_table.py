import sys
import click
import pandas as pd

from conf import DATASET_PRETTY_NAMES, WEIGHT_PRETTY_NAMES


@click.command()
@click.argument("result", type=click.File('r'))
@click.argument("table", type=click.File('w'))
@click.option("--value", '-v', default="error")
@click.option("--index", '-i', default="name")
@click.option("--label", '-l', default="tab:results")
def main(result, table, value, index, label):

    baseline = "uniform"

    data = pd.read_csv(result, index_col=0).set_index(["weight", "seed"])

    # data = data.assign(error=1.0-data["acc"])
    # data.drop(columns=["dataset_seed", "acc"], inplace=True)
    data.drop(columns="dataset_seed", inplace=True)

    data_baseline = data.query(f"weight == '{baseline}'") \
                        .reset_index(level="weight", drop=True)

    data_rel = data.divide(data_baseline, axis="index", level="seed") \
                   .rename(columns={"error": "error_relative"})
    data_rel = data_rel.assign(error_relative_change=1.0 - data_rel.error_relative)

    data_new = pd.concat([data, data_rel], axis="columns", join="inner")
    data_new.reset_index(inplace=True)
    data_new.replace({"weight": WEIGHT_PRETTY_NAMES}, inplace=True)

    # d = data_new.reset_index().replace({"weight": WEIGHT_PRETTY_NAMES})
    # data.replace({"name": DATASET_PRETTY_NAMES}, inplace=True)

    columns = ["mean", "std"]
    summary = data_new.groupby("weight").describe()

    # # summary = summary.reset_index() \
    # #                  .pivot(index=index, columns="weight", values=columns)

    table.write(summary.to_latex(
        columns=pd.MultiIndex.from_product([["error", "error_relative_change"],
                                            columns]),
        float_format="{:0.3f}".format,
        caption=f"{value} across 10 trials.", label=label,
        formatters={
            ("error", "std"): r"($\pm${:0.2f})".format,
            ("error_relative_change", "std"): r"($\pm${:0.2f})".format
        },
        escape=False)
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
