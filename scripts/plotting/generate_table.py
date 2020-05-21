import sys
import click
import pandas as pd

from conf import WEIGHT_PRETTY_NAMES


@click.command()
@click.argument("result", type=click.File('r'))
@click.argument("table", type=click.File('w'))
@click.option("--value", '-v', default="error")
@click.option("--index", '-i', default="name")
@click.option("--label", '-l', default="tab:results")
def main(result, table, value, index, label):

    data = pd.read_csv(result, index_col=0)
    data = data.assign(error=1.0-data["acc"])
    data.replace({"weight": WEIGHT_PRETTY_NAMES}, inplace=True)

    columns = ["mean", "std"]
    summary = data.groupby("weight")[value].describe()  # [columns]

    # summary = data.groupby([index, "weight"])[value].describe().reset_index() \
    #               .pivot(index=index, columns="weight", values=["mean", "std"])

    table.write(summary.to_latex(
        columns=columns,
        float_format="{:0.3f}".format,
        caption=f"{value} across 10 trials.", label=label,
        formatters={"std": r"($\pm${:0.3f})".format}, escape=False)
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
