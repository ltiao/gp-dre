import sys
import click

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from gpdre import DensityRatioMarginals, GaussianProcessDensityRatioEstimator
from gpdre.base import LogisticRegressionDensityRatioEstimator
from gpdre.external.rulsif import RuLSIFDensityRatioEstimator
from gpdre.external.kliep import KLIEPDensityRatioEstimator
from gpdre.external.kmm import KMMDensityRatioEstimator
from gpdre.datasets import make_classification_dataset
from gpdre.plotting import fill_between_stddev

from gpflow.models import VGP
from gpflow.kernels import SquaredExponential, Matern52
from gpflow.optimizers import Scipy

from sklearn.linear_model import LinearRegression

from pathlib import Path

# shortcuts
tfd = tfp.distributions

GOLDEN_RATIO = 0.5 * (1 + np.sqrt(5))
WIDTH = 397.48499
OUTPUT_DIR = "logs/figures/"

SEED = 8888
DATASET_SEED = 8888
RATE = 0.5

JITTER = 1e-6

num_samples = 800
num_index_points = 512
num_output_points = 512
num_features = 1
num_mc_samples = 10

kernel_cls = SquaredExponential
optimizer = Scipy()


def pt_to_in(x):

    pt_per_in = 72.27
    return x / pt_per_in


def size(width, aspect=GOLDEN_RATIO):

    width_in = pt_to_in(width)
    return (width_in, width_in / aspect)


def poly(x):
    return - x + x**3


@click.command()
@click.argument("name")
@click.option('--width', '-w', type=float, default=WIDTH)
@click.option('--aspect', '-a', type=float, default=GOLDEN_RATIO)
@click.option('--extension', '-e', multiple=True, default=["png"])
@click.option("--output-dir", default=OUTPUT_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory.")
def main(name, width, aspect, extension, output_dir):

    figsize = size(width, aspect)
    suffix = f"{width:.0f}x{width/aspect:.0f}"

    # preamble
    rc = {
        "figure.figsize": figsize,
        "font.serif": ["Times New Roman"],
        "text.usetex": True,
    }

    sns.set(context="paper",
            style="ticks",
            palette="colorblind",
            font="serif",
            rc=rc)

    base_path = Path(output_dir).joinpath(name)
    # /preamble

    p = tfd.Normal(loc=0.0, scale=1.0)
    qs = {
        "same": tfd.Normal(loc=0.0, scale=1.0),
        "scale_lesser": tfd.Normal(loc=0.0, scale=tf.sqrt(0.6)),
        "scale_greater": tfd.Normal(loc=0.0, scale=tf.sqrt(2.0)),
        "loc_different": tfd.Normal(loc=0.5, scale=1.0),
        "additive": tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=[0.95, 0.05]),
            components_distribution=tfd.Normal(loc=[0.0, 3.0], scale=[1.0, 1.0])),
        "bimodal": tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=[0.4, 0.6]),
            components_distribution=tfd.Normal(loc=[2.0, -3.0], scale=[0.5, 1.0]))
    }

    xlims = {
        "same": (-5.0, 5.0),
        "scale_lesser": (-5.0, 5.0),
        "scale_greater": (-5.0, 5.0),
        "loc_different": (-5.0, 5.0),
        "additive": (-5.0, 5.0),
        "bimodal": (-5.0, 5.0)
    }

    rmaxs = {
        "same": 3.0,
        "scale_lesser": 3.0,
        "scale_greater": 3.0,
        "loc_different": 3.0,
        "additive": 3.0,
        "bimodal": 250.0
    }

    for q_key in qs:

        print(q_key)

        output_path = base_path.joinpath(q_key)
        output_path.mkdir(parents=True, exist_ok=True)

        q = qs[q_key]

        xmin, xmax = xlims[q_key]
        rmax = rmaxs[q_key]

        X_grid = np.linspace(xmin, xmax, num_index_points).reshape(-1, num_features)
        R_grid = np.linspace(0.0, rmax, num_output_points).reshape(-1, 1)
        x_grid, r_grid = np.meshgrid(X_grid, R_grid)

        r = DensityRatioMarginals(top=p, bot=q)
        X_top, X_bot = r.make_dataset(num_samples, rate=RATE, seed=DATASET_SEED)
        X, s = make_classification_dataset(X_top, X_bot)

        # Figure 1
        fig, ax = plt.subplots()

        ax.plot(X_grid, r.bot.prob(X_grid), c="tab:orange",
                label=r"$p_{\mathrm{tr}}(\mathbf{x})$")
        ax.plot(X_grid, r.top.prob(X_grid), c="tab:purple",
                label=r"$p_{\mathrm{te}}(\mathbf{x})$")

        ax.set_xlabel('$x$')
        ax.set_ylabel('density')

        ax.legend()

        for ext in extension:
            fig.savefig(output_path.joinpath(f"distribution_{suffix}.{ext}"),
                        bbox_inches="tight")

        plt.show()

        # Figure 3
        fig, ax = plt.subplots()

        ax.plot(X_grid, r.logit(X_grid), c='k',
                label=r"$f(x) = \log p(x) - \log q(x)$")

        ax.set_xlabel('$x$')
        ax.set_ylabel('$f(x)$')

        ax.legend()

        for ext in extension:
            fig.savefig(output_path.joinpath(f"logit_{suffix}.{ext}"),
                        bbox_inches="tight")

        plt.show()

        # Figure 4
        fig, ax = plt.subplots()

        ax.plot(X_grid, r.ratio(X_grid), c='k', label=r"$r(x) = \exp{f(x)}$")

        ax.set_xlabel('$x$')
        ax.set_ylabel('$r(x)$')

        ax.legend()

        ax.set_ylim(0.0, rmax)

        for ext in extension:
            fig.savefig(output_path.joinpath(f"ratio_{suffix}.{ext}"),
                        bbox_inches="tight")

        plt.show()

        gpdre = GaussianProcessDensityRatioEstimator(input_dim=num_features,
                                                     kernel_cls=kernel_cls,
                                                     vgp_cls=VGP,
                                                     jitter=JITTER,
                                                     seed=SEED)
        gpdre.compile(optimizer=optimizer)
        gpdre.fit(X_top, X_bot)

        log_ratio_mean = gpdre.logit(X_grid, convert_to_tensor_fn=tfd.Distribution.mean)
        log_ratio_stddev = gpdre.logit(X_grid, convert_to_tensor_fn=tfd.Distribution.stddev)

        fig, ax = plt.subplots()

        ax.plot(X_grid, r.logit(X_grid), c='k',
                label=r"$f(x) = \log p(x) - \log q(x)$")

        ax.plot(X_grid, log_ratio_mean.numpy().T,
                label="posterior mean")
        fill_between_stddev(X_grid.squeeze(),
                            log_ratio_mean.numpy().squeeze(),
                            log_ratio_stddev.numpy().squeeze(), alpha=0.1,
                            label="posterior std dev", ax=ax)

        ax.set_xlabel('$x$')
        ax.set_ylabel('$f(x)$')

        ax.legend()

        for ext in extension:
            fig.savefig(output_path.joinpath(f"logit_posterior_{suffix}.{ext}"),
                        bbox_inches="tight")

        plt.show()

        ratio = gpdre.ratio_distribution(X_grid)

        fig, ax = plt.subplots()

        ax.plot(X_grid, r(X_grid), c='k', label=r"$r(x) = \exp{f(x)}$")

        ax.plot(X_grid, ratio.distribution.quantile(0.5),
                c="tab:blue", label="median")

        ax.fill_between(X_grid.squeeze(),
                        ratio.distribution.quantile(0.25),
                        ratio.distribution.quantile(0.75),
                        alpha=0.1, label="interquartile range")

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$r(x)$")

        ax.set_ylim(0.0, rmax)

        ax.legend()

        for ext in extension:
            fig.savefig(output_path.joinpath(f"ratio_posterior_quantiles_{suffix}.{ext}"),
                        bbox_inches="tight")

        plt.show()

        fig, ax = plt.subplots()

        ax.plot(X_grid, r(X_grid), c='k', label=r"$r(x) = \exp{f(x)}$")
        ax.plot(X_grid, ratio.distribution.quantile(0.5),
                c="tab:blue", label="median")
        ax.fill_between(X_grid.squeeze(),
                        ratio.distribution.quantile(0.25),
                        ratio.distribution.quantile(0.75),
                        alpha=0.1, label="interquartile range")

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$r(x)$")

        ax.set_ylim(0.0, rmax)

        ax.legend()

        for ext in extension:
            fig.savefig(output_path.joinpath(f"ratio_posterior_quantiles_{suffix}.{ext}"),
                        bbox_inches="tight")

        plt.show()

        ratio_marginal = gpdre.ratio_distribution(X_grid,
                                                  reinterpreted_batch_ndims=None)

        fig, ax = plt.subplots()

        contours = ax.pcolormesh(x_grid, r_grid, ratio_marginal.prob(R_grid),
                                 zorder=-15, vmax=1.0, cmap="Blues")
        ax.set_rasterization_zorder(-10)

        ax.plot(X_grid, r(X_grid), c='k', label=r"$r(x) = \exp{f(x)}$")

        fig.colorbar(contours, extend="max", ax=ax)

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$r(x)$")

        ax.set_ylim(0.0, rmax)

        ax.legend()

        for ext in extension:
            fig.savefig(output_path.joinpath(f"ratio_posterior_density_{suffix}.{ext}"),
                        bbox_inches="tight")

        plt.show()

        fig, ax = plt.subplots()

        contours = ax.pcolormesh(x_grid, r_grid, ratio_marginal.prob(R_grid),
                                 zorder=-15, vmax=1.0, cmap="Blues")
        ax.set_rasterization_zorder(-10)

        ax.plot(X_grid, r(X_grid), c='k', label=r"$r(x) = \exp{f(x)}$")

        ax.plot(X_grid, ratio_marginal.mode(), c="tab:blue",
                label="posterior mode")
        ax.plot(X_grid, ratio_marginal.mean(), c="tab:blue", linestyle="--",
                label="posterior mean")
        ax.plot(X_grid, gpdre.ratio(X_grid, convert_to_tensor_fn=lambda d: d.distribution.quantile(0.5)),
                c="tab:blue", linestyle=':', label="posterior median")

        fig.colorbar(contours, extend="max", ax=ax)

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$r(x)$")

        ax.set_ylim(0.0, rmax)

        ax.legend()

        for ext in extension:
            fig.savefig(output_path.joinpath(f"ratio_posterior_mode_{suffix}.{ext}"),
                        bbox_inches="tight")

        plt.show()

        # Figure 5
        fig, ax = plt.subplots()

        ax.plot(X_grid, gpdre.conditional(X_grid, num_mc_samples).mean().numpy().T,
                color="tab:blue", linewidth=0.4, alpha=0.6)
        ax.plot(X_grid, r.prob(X_grid), c='k',
                label=r"$\pi(x) = \sigma(f(x))$")
        ax.scatter(X, s, c=s, s=12.**2, marker='s', alpha=0.1, cmap="PuOr")
        ax.set_yticks([0, 1])
        ax.set_yticklabels([r"$x_q \sim q(x)$", r"$x_p \sim p(x)$"])
        ax.set_xlabel('$x$')

        ax.legend()

        for ext in extension:
            fig.savefig(output_path.joinpath(f"prob_{suffix}.{ext}"),
                        bbox_inches="tight")

        plt.show()

        # Baselines
        r_kliep = KLIEPDensityRatioEstimator(seed=SEED)
        r_kliep.fit(X_top, X_bot)

        r_rulsif = RuLSIFDensityRatioEstimator(alpha=1e-2)
        r_rulsif.fit(X_top, X_bot)

        r_kmm = KMMDensityRatioEstimator(B=1000.0)
        r_kmm.fit(X_top, X_bot)

        r_logreg = LogisticRegressionDensityRatioEstimator(seed=SEED)
        r_logreg.fit(X_top, X_bot)

        fig, ax = plt.subplots()

        ax.plot(X_grid, r(X_grid), c='k', label=r"$r(x) = \exp{f(x)}$")
        ax.plot(X_grid, r_kliep.ratio(X_grid), label="KLIEP")
        ax.plot(X_grid, r_rulsif.ratio(X_grid), label="RuLSIF")
        ax.plot(X_grid, r_logreg.ratio(X_grid), label="LogReg")

        ax.scatter(X_bot, r_kmm.ratio(X_bot), alpha=0.6, label="KMM")

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$r(x)$")
        ax.set_ylim(0, rmax)

        ax.legend()

        for ext in extension:
            fig.savefig(output_path.joinpath(f"ratio_baselines_{suffix}.{ext}"),
                        bbox_inches="tight")

        plt.show()

    # x0 = -0.5
    # x1 = 0.25

    # r0_min, r0_max = 0, 1.55  # 0.21
    # r1_min, r1_max = 0, 1.55

    # Y_grid = np.linspace(r_min, r_max, y_num_points).reshape(-1, 1)
    # x, y = np.meshgrid(X_grid, Y_grid)

    # fig = plt.figure(constrained_layout=True)
    # spec = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)

    # ax1 = fig.add_subplot(spec[0, 0])
    # ax2 = fig.add_subplot(spec[0, 1], sharey=ax1)

    # r0 = gpdre.ratio_distribution(np.float64([[x0]]),
    #                               reinterpreted_batch_ndims=None)

    # ax1.plot(Y_grid, r0.prob(Y_grid))
    # ax1.vlines(r(x0).numpy(), r0_min, r0_max, linewidth=1.0)
    # ax1.vlines(r0.mode(), r0_min, r0_max,
    #            color="tab:blue", linewidth=1.0)
    # ax1.vlines(r0.mean(), r0_min, r0_max,
    #            color="tab:blue", linestyle="--", linewidth=1.0)

    # ax1.set_ylabel(r"$q(r_0)$")
    # ax1.set_xlabel(r"$r(x_0)$")

    # ax1.set_ylim(r0_min, r0_max)

    # r1 = gpdre.ratio_distribution(np.float64([[x1]]),
    #                               reinterpreted_batch_ndims=None)
    # ax2.plot(Y_grid, r1.prob(Y_grid))
    # ax2.vlines(r(x1).numpy(), r1_min, r1_max, linewidth=1.0)
    # ax2.vlines(r1.mode(), r1_min, r1_max,
    #            color="tab:blue", linewidth=1.0)
    # ax2.vlines(r1.mean(), r1_min, r1_max,
    #            color="tab:blue", linestyle="--", linewidth=1.0)

    # ax2.set_ylabel(r"$q(r_1)$")
    # ax2.set_xlabel(r"$r(x_1)$")

    # ax2.set_ylim(r1_min, r1_max)

    # ax3 = fig.add_subplot(spec[1:, :])

    # ax3.vlines(x0, r_min, r_max, color='tab:orange', linewidth=1.0)
    # ax3.vlines(x1, r_min, r_max, color='tab:orange', linewidth=1.0)

    # ax3.text(x0 + 0.05, r_max - 0.5, rf"$x_0 = {x0}$", color='tab:orange')
    # ax3.text(x1 + 0.05, r_max - 0.5, rf"$x_1 = {x1}$", color='tab:orange')

    # contours = ax3.pcolormesh(x, y, ratio_marginal.prob(Y_grid),
    #                           zorder=-15, vmax=1.0, cmap="Blues")
    # ax3.set_rasterization_zorder(-10)

    # ax3.plot(X_grid, r(X_grid), c='k', label=r"$r(x) = \exp{f(x)}$")

    # ax3.plot(X_grid, ratio_marginal.mode(), c="tab:blue",
    #          label="posterior mode")
    # ax3.plot(X_grid, ratio_marginal.mean(), c="tab:blue", linestyle="--",
    #          label="posterior mean")

    # # fig.colorbar(contours, extend="max", ax=[ax2, ax3])

    # ax3.set_xlabel(r"$x$")
    # ax3.set_ylabel(r"$r(x)$")

    # ax3.set_ylim(r_min, r_max)

    # ax3.legend()

    # for ext in extension:
    #     fig.savefig(output_path.joinpath(f"teaser_{suffix}.{ext}"),
    #                 bbox_inches="tight")

    # plt.show()

    # model_uniform_train = LinearRegression().fit(X_train, y_train)
    # model_uniform_test = LinearRegression().fit(X_test, y_test)
    # model_exact_train = LinearRegression().fit(X_train, y_train,
    #                                            sample_weight=r.ratio(X_train) \
    #                                                           .numpy().squeeze())
    # sample_weight = gpdre.ratio(X_train, convert_to_tensor_fn=tfd.Distribution.mode).numpy().squeeze()
    # model_mode_train = LinearRegression().fit(X_train, y_train,
    #                                           sample_weight=sample_weight)

    # frames = []
    # frames.append(pd.DataFrame(dict(name="training (unweighted)",
    #                                 x=np.squeeze(X_grid, axis=-1),
    #                                 y=model_uniform_train.predict(X_grid))))
    # frames.append(pd.DataFrame(dict(name="test (unweighted)",
    #                                 x=np.squeeze(X_grid, axis=-1),
    #                                 y=model_uniform_test.predict(X_grid))))
    # frames.append(pd.DataFrame(dict(name="training (exact)",
    #                                 x=np.squeeze(X_grid, axis=-1),
    #                                 y=model_exact_train.predict(X_grid))))
    # frames.append(pd.DataFrame(dict(name="training (LGP mode)",
    #                                 x=np.squeeze(X_grid, axis=-1),
    #                                 y=model_mode_train.predict(X_grid))))
    # frames.append(pd.DataFrame(dict(name="true",
    #                                 x=np.squeeze(X_grid, axis=-1),
    #                                 y=np.squeeze(poly(X_grid), axis=-1))))
    # data = pd.concat(frames, axis="index", ignore_index=True, sort=True)

    # # Figure 7
    # fig, ax = plt.subplots()

    # sns.lineplot(x='x', y='y', hue="name", style="name", data=data, ax=ax)

    # ax.scatter(X_train, y_train, alpha=0.6, label="training")
    # ax.scatter(X_test, y_test, marker='x', alpha=0.6, label="test")

    # ax.set_xlabel(r'$x$')
    # ax.set_ylabel(r'$y$')

    # ax.set_xlim(-0.5, 1.5)
    # ax.set_ylim(-1.0, 1.5)

    # ax.legend(loc="upper left")

    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[1:], labels[1:], title=None,  # fontsize="small",
    #           handletextpad=0.5, columnspacing=1,
    #           ncol=2, frameon=True, loc="upper left")

    # for ext in extension:
    #     fig.savefig(output_path.joinpath(f"dataset_{suffix}.{ext}"),
    #                 bbox_inches="tight")

    # plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
