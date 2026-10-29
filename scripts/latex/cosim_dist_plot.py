"""Cosine-similarity distribution plots for AugNet / FMap embeddings.

Loads pre-generated relationship pickle files (from gen_dataset_carefully.py or
noisy_rerank_plot.py) for a target cluster and plots the positive vs. negative
cosine-similarity distributions using a publication-ready matplotlib style.

The GNN used is the one trained on cluster 0 (the checkpoint tag does not
encode a cluster index, so it is shared across clusters by construction).

Usage
-----
# AugNet, K=40 noiseless, cluster 2
python -B scripts/latex/cosim_dist_plot.py --dataset ptc_fr --cluster 2 --ratio 0.13

# AugNet, multiple sigmas
python -B scripts/latex/cosim_dist_plot.py --dataset ptc_fr --cluster 2 --ratio 0.13 \\
    --sigmas 0 0.5 1

# FMap mode
python -B scripts/latex/cosim_dist_plot.py --dataset ptc_fr --cluster 2 --ratio 0.13 \\
    --use-fmap --fmap-loss ce-avg --estol 1e-5

# Random relationships
python -B scripts/latex/cosim_dist_plot.py --dataset ptc_fr --cluster 2 --ratio 0.13 \\
    --mode random
"""

import argparse
import os
import pickle
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.legend_handler import HandlerBase
import numpy as np
from scipy.stats import gaussian_kde
import torch.nn.functional as F

# ── project root on path ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "misc"))
os.chdir(ROOT)

from wandb_to_human import convert_name
from reform.utils.script_utils import (
    K_VAL,
    ratio_suffix as _ratio_suffix,
    add_aug_args, build_runner, load_aug_checkpoint,
    add_fmap_args, build_fmap_runner, compute_fmap_train_embeddings,
    compute_train_embeddings, cosine_sims_for_relationships,
)

# ── matplotlib style ──────────────────────────────────────────────────────────
STYLE = {
    "text.usetex":              True,
    "text.latex.preamble":      r"\boldmath",
    "font.family":              "serif",
    "font.size":                11,
    "axes.labelsize":           21,
    "axes.titlesize":           20,
    "xtick.labelsize":          18,
    "ytick.labelsize":          18,
    "axes.spines.top":          True,
    "axes.spines.right":        True,
    "figure.dpi":               150,
}

POS_COLOR = "#1558a0"
NEG_COLOR = "#c0392b"

KDE_POINTS = 400   # x-grid resolution for KDE curves

SIGMA_POS_COLORS = {
    0.1: "#17becf",   # cyan
    0.5: "#ff7f0e",   # orange
    1:   "#2ca02c",   # green
    10:  "#9467bd",   # purple
}

COMBINED_NEG_COLOR = "#c0392b"   # single red for the shared negative distribution


class _HandlerBlank(HandlerBase):
    """Legend handler that renders no handle — for text-only prefix entries."""
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        handlebox.set_width(0)
        return None


# ── embedding computation ─────────────────────────────────────────────────────

def _get_cosine_sims(cluster_runner):
    query_aug_embs, corpus_aug_embs, q_gts = compute_train_embeddings(cluster_runner)
    cosine_sims = (
        F.normalize(query_aug_embs[:, 0, :], dim=-1) @
        F.normalize(corpus_aug_embs[:, 0, :], dim=-1).T
    ).cpu()
    return cosine_sims, q_gts


def _split_by_gt(cosine_sims, dataset, cluster_idx):
    """Split a precomputed cosine_sims matrix by true ground truth labels."""
    import torch
    gt_path = f"data/{dataset}/preprocessed/splits/train/clusters/{cluster_idx}/all_gt.pt"
    all_gt = torch.load(gt_path).reshape(cosine_sims.shape[0], cosine_sims.shape[1]).bool()
    pos_vals = cosine_sims[all_gt].tolist()
    neg_vals = cosine_sims[~all_gt].tolist()
    return pos_vals, neg_vals


def _get_cosine_sims_fmap(fmap_runner):
    query_aug_embs, corpus_aug_embs, q_gts = compute_fmap_train_embeddings(fmap_runner)
    cosine_sims = (
        F.normalize(query_aug_embs[:, 0, :], dim=-1) @
        F.normalize(corpus_aug_embs[:, 0, :], dim=-1).T
    ).cpu()
    return cosine_sims, q_gts


# ── relationship loading ──────────────────────────────────────────────────────

def _lsh_pkl_path(out_base, cluster_idx, augnet_name, K_val, ratio, sigma):
    base  = f"all_corpus_negs_K={K_val}{_ratio_suffix(ratio)}"
    noise = "" if sigma == 0 else f"_noise_sigma={sigma}"
    return os.path.join(out_base, str(cluster_idx), augnet_name,
                        f"lsh_relationships_{base}{noise}.pkl")


def _random_pkl_path(out_base, cluster_idx, K_val, ratio):
    suffix = f"all_corpus_negs_K={K_val}{_ratio_suffix(ratio)}"
    return os.path.join(out_base, str(cluster_idx), f"random_relationships_{suffix}.pkl")


# ── plotting ──────────────────────────────────────────────────────────────────

def _kde_curve(vals, x):
    """Evaluate a Gaussian KDE of vals on the grid x."""
    return gaussian_kde(vals)(x)


def _plot_kde(ax, vals, color, alpha_fill=0.40, linewidth=1.6, linestyle="-", zorder=2):
    x = np.linspace(min(vals) - 0.05, max(vals) + 0.05, KDE_POINTS)
    y = _kde_curve(vals, x)
    ax.fill_between(x, y, alpha=alpha_fill, color=color, zorder=zorder)
    ax.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle, zorder=zorder + 1)
    return x, y


def plot_distribution(
    pos_vals, neg_vals,
    cluster_idx, sigma_or_label,
    out_path, K_val,
    is_random=False,
    show_ylabel=False,
    approx_counts=False,  # retained for call-site compatibility; unused
    show_legend=True,
):
    pos_vals = (np.array(pos_vals) + 1) / 2
    neg_vals = (np.array(neg_vals) + 1) / 2

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(3.5, 2.8))

        x_pos, y_pos = _plot_kde(ax, pos_vals, POS_COLOR)
        x_neg, y_neg = _plot_kde(ax, neg_vals, NEG_COLOR)

        ax.set_xlabel(r"\textbf{Score} $\rightarrow$")
        ax.set_ylabel(r"\textbf{Density} $\rightarrow$" if show_ylabel else "")

        if sigma_or_label != "GT":
            title = r"\textbf{Random}" if is_random else \
                    r"\textbf{$\sigma = 0$}" if sigma_or_label == 0 else \
                    rf"\textbf{{$\sigma = {sigma_or_label}$}}"
            ax.set_title(title)
            if show_legend:
                handles = [
                    mpatches.Patch(facecolor=POS_COLOR, edgecolor="none", label="Positive"),
                    mpatches.Patch(facecolor=NEG_COLOR, edgecolor="none", label="Negative"),
                ]
                ax.legend(handles=handles, loc="upper right", fontsize=10,
                          frameon=True, framealpha=0.85, edgecolor="#aaaaaa")
        else:
            pos_pk_idx = int(np.argmax(y_pos))
            neg_pk_idx = int(np.argmax(y_neg))
            # right-tail anchor: where positive drops to 35% of its peak
            pos_right_idx = pos_pk_idx + int(
                np.argmin(np.abs(y_pos[pos_pk_idx:] - y_pos[pos_pk_idx] * 0.35))
            )
            ax.annotate(
                r"\textbf{Negative}",
                xy=(x_neg[neg_pk_idx], y_neg[neg_pk_idx] * 0.9),
                xycoords="data",
                xytext=(0.55, 0.82),
                textcoords="axes fraction",
                fontsize=13, color=NEG_COLOR, ha="center",
                arrowprops=dict(arrowstyle="->", color=NEG_COLOR, lw=0.8,
                                connectionstyle="arc3,rad=-0.15"),
            )
            ax.annotate(
                r"\textbf{Positive}",
                xy=(x_pos[pos_right_idx], y_pos[pos_right_idx]),
                xycoords="data",
                xytext=(0.78, 0.45),
                textcoords="axes fraction",
                fontsize=13, color=POS_COLOR, ha="center",
                arrowprops=dict(arrowstyle="->", color=POS_COLOR, lw=0.8,
                                connectionstyle="arc3,rad=0.25"),
            )

        if sigma_or_label == "GT":
            data_max = float(max(pos_vals.max(), neg_vals.max()))
            last_tick = round(min(0.75, data_max), 2)
            half_tick = round(last_tick / 2, 2)
            ax.set_xlim(right=last_tick + 0.02)
            ax.set_xticks([0.0, half_tick, last_tick])
            ax.xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, p: f"${x:.2f}$"))
        else:
            ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])

        fig.subplots_adjust(left=0.26, right=0.95, bottom=0.26, top=0.93)
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  Saved: {out_path}")


def plot_combined_distribution(sigma_data, out_path, K_val, show_ylabel=False, show_legend=True):
    """Overlay smooth KDE curves for multiple sigmas on one axes.

    sigma_data : list of (sigma, pos_vals, neg_vals) in display order
    Positive = one distinct colour per sigma; negative = single red (first sigma only,
    since all sigma negatives overlap).
    """
    sigma_data_t = [
        (sigma, (np.array(pos) + 1) / 2, (np.array(neg) + 1) / 2)
        for sigma, pos, neg in sigma_data
    ]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(3.5, 2.8))

        _, _, neg_vals_ref = sigma_data_t[0]
        _plot_kde(ax, neg_vals_ref, COMBINED_NEG_COLOR, alpha_fill=0.45, zorder=2)

        sigma_handles, sigma_labels = [], []
        for i, (sigma, pos_vals, _) in enumerate(sigma_data_t):
            pos_color = SIGMA_POS_COLORS[sigma]
            s = int(sigma) if float(sigma) == int(float(sigma)) else sigma
            _plot_kde(ax, pos_vals, pos_color, alpha_fill=0.45, zorder=3 + i)
            sigma_handles.append(mpatches.Patch(facecolor=pos_color, edgecolor="none"))
            sigma_labels.append(rf"${s}$")

        ax.set_xlabel(r"\textbf{Score} $\rightarrow$")
        ax.set_ylabel(r"\textbf{Density} $\rightarrow$")

        if show_legend:
            prefix = mpatches.Patch(alpha=0)
            leg = ax.legend(
                [prefix] + sigma_handles,
                [r"$\sigma \rightarrow$"] + sigma_labels,
                handler_map={prefix: _HandlerBlank()},
                loc="upper center", fontsize=9,
                handlelength=1.2, handleheight=0.8, handletextpad=0.4,
                columnspacing=0.6, ncol=len(sigma_handles) + 1,
                frameon=True, framealpha=0.5, edgecolor="#aaaaaa",
            )
            leg.get_frame().set_linewidth(0.8)

        data_max = float(max(
            max(pos.max() for _, pos, _ in sigma_data_t),
            neg_vals_ref.max(),
        ))
        last_tick = round(min(0.75, data_max), 2)
        half_tick = round(last_tick / 2, 2)
        ax.set_xlim(right=last_tick + 0.02)
        ax.set_xticks([0.0, half_tick, last_tick])
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: f"${x:.2f}$"))

        fig.subplots_adjust(left=0.26, right=0.95, bottom=0.26, top=0.93)
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  Saved: {out_path}")


def save_color_legend(out_path):
    """Standalone legend: single row — σ → prefix followed by σ=0.1/0.5/1/10 boxes."""
    sigma_handles, sigma_labels = [], []
    for sigma in [0.1, 0.5, 1, 10]:
        s = int(sigma) if float(sigma) == int(float(sigma)) else sigma
        sigma_handles.append(mpatches.Patch(facecolor=SIGMA_POS_COLORS[sigma], edgecolor="none"))
        sigma_labels.append(rf"${s}$")

    prefix = mpatches.Patch(alpha=0)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.2, 0.5))
        ax.axis("off")
        ax.legend(
            handles=[prefix] + sigma_handles,
            labels=[r"$\sigma \rightarrow$"] + sigma_labels,
            handler_map={prefix: _HandlerBlank()},
            loc="center",
            ncol=len(sigma_handles) + 1,
            frameon=False,
            fontsize=10,
            handlelength=1.2,
            handleheight=0.9,
            handletextpad=0.4,
            columnspacing=0.8,
        )
        fig.tight_layout(pad=0)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    result = subprocess.run(
        ["pdfcrop", out_path, out_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  Warning: pdfcrop failed — {result.stderr.strip()}")
    print(f"  Saved colour legend: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Publication-quality cosine-sim distribution plots for AugNet/FMap."
    )
    parser.add_argument("--dataset",   type=str, default=None)
    parser.add_argument("--cluster",   type=int, default=None,
                        help="Evaluation cluster (1-4).")
    parser.add_argument("--gpu",       type=str, default=None)
    parser.add_argument("--save-legend", action="store_true", default=False,
                        help="Write standalone color legend PDF and exit (no model needed)")
    parser.add_argument("--mode",      type=str, default="lsh",
                        choices=["lsh", "combined", "random", "gt"])
    parser.add_argument("--sigmas",
                        type=lambda x: int(x) if float(x) == int(float(x)) else float(x),
                        nargs="+", default=[0, 0.5, 1],
                        help="[LSH] Sigma values to load (default: 0 0.5 1)")
    parser.add_argument("--ratio",     type=float, default=None)
    parser.add_argument("--K_val",     type=int, default=K_VAL)
    parser.add_argument("--out_plots", type=str, default="plots/cosim_dist")
    parser.add_argument("--show-ylabel", action="store_true", default=False,
                        help="Show Y-axis Density label (default: off; pass for sigma=0.1 plot)")
    parser.add_argument("--no-legend",  action="store_true", default=False,
                        help="Omit in-plot legend; appends _noleg to output filenames")
    parser.add_argument("--use-fmap",  action="store_true", default=False)
    add_aug_args(parser)
    add_fmap_args(parser)
    args = parser.parse_args()

    if (args.no_legend or args.save_legend) and args.out_plots == parser.get_default("out_plots"):
        args.out_plots = os.path.join("plots", "appendix", "cosim_dist")

    if args.save_legend:
        os.makedirs(args.out_plots, exist_ok=True)
        save_color_legend(os.path.join(args.out_plots, "cosim_color_legend.pdf"))
        return

    if args.dataset is None or args.cluster is None:
        parser.error("--dataset and --cluster are required unless --save-legend is used")

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    out_base = f"data/{args.dataset}/preprocessed/splits/train/clusters"
    os.makedirs(args.out_plots, exist_ok=True)

    # Build runner and compute embeddings
    if args.use_fmap:
        runner      = build_fmap_runner(args)
        augnet_name = convert_name(runner.conf.task.name)
        cosine_sims, q_gts = _get_cosine_sims_fmap(runner)
    else:
        runner      = build_runner(args)
        augnet_name = convert_name(runner.conf.task.name)
        load_aug_checkpoint(runner)
        cosine_sims, q_gts = _get_cosine_sims(runner)

    ratio_sfx  = _ratio_suffix(args.ratio)
    noleg_sfx  = "_noleg" if args.no_legend else ""
    show_leg   = not args.no_legend

    if args.mode == "gt":
        pos_vals, neg_vals = _split_by_gt(cosine_sims, args.dataset, args.cluster)
        print(
            f"  ground truth: pos n={len(pos_vals)} mean={np.mean(pos_vals):.4f} | "
            f"neg n={len(neg_vals)} mean={np.mean(neg_vals):.4f}"
        )
        fname = (
            f"cosim_{args.dataset}_K={args.K_val}{ratio_sfx}"
            f"_cluster{args.cluster}_gt_{augnet_name}{noleg_sfx}.pdf"
        )
        plot_distribution(
            pos_vals, neg_vals,
            args.cluster, "GT",
            os.path.join(args.out_plots, fname),
            args.K_val, show_ylabel=args.show_ylabel, approx_counts=True,
            show_legend=show_leg,
        )
    elif args.mode == "random":
        pkl_path = _random_pkl_path(out_base, args.cluster, args.K_val, args.ratio)
        print(f"Loading {pkl_path}")
        with open(pkl_path, "rb") as f:
            relationships = pickle.load(f)
        pos_vals, neg_vals, _ = cosine_sims_for_relationships(
            relationships, cosine_sims, q_gts
        )
        fname = (
            f"cosim_{args.dataset}_K={args.K_val}{ratio_sfx}"
            f"_cluster{args.cluster}_random_{augnet_name}{noleg_sfx}.pdf"
        )
        plot_distribution(
            pos_vals, neg_vals,
            args.cluster, None,
            os.path.join(args.out_plots, fname),
            args.K_val, is_random=True, show_ylabel=args.show_ylabel,
            show_legend=show_leg,
        )
    elif args.mode == "combined":
        sigma_data = []
        for sigma in args.sigmas:
            pkl_path = _lsh_pkl_path(
                out_base, args.cluster, augnet_name, args.K_val, args.ratio, sigma
            )
            print(f"Loading {pkl_path}")
            with open(pkl_path, "rb") as f:
                relationships = pickle.load(f)
            pos_vals, neg_vals, _ = cosine_sims_for_relationships(
                relationships, cosine_sims, q_gts
            )
            print(
                f"  sigma={sigma}: pos n={len(pos_vals)} mean={np.mean(pos_vals):.4f} | "
                f"neg n={len(neg_vals)} mean={np.mean(neg_vals):.4f}"
            )
            sigma_data.append((sigma, pos_vals, neg_vals))
        fname = (
            f"cosim_{args.dataset}_K={args.K_val}{ratio_sfx}"
            f"_cluster{args.cluster}_combined_{augnet_name}{noleg_sfx}.pdf"
        )
        plot_combined_distribution(
            sigma_data,
            os.path.join(args.out_plots, fname),
            args.K_val,
            show_ylabel=args.show_ylabel,
            show_legend=show_leg,
        )
    else:
        for i, sigma in enumerate(args.sigmas):
            pkl_path = _lsh_pkl_path(
                out_base, args.cluster, augnet_name, args.K_val, args.ratio, sigma
            )
            print(f"Loading {pkl_path}")
            with open(pkl_path, "rb") as f:
                relationships = pickle.load(f)
            pos_vals, neg_vals, _ = cosine_sims_for_relationships(
                relationships, cosine_sims, q_gts
            )
            print(
                f"  sigma={sigma}: pos n={len(pos_vals)} mean={np.mean(pos_vals):.4f} | "
                f"neg n={len(neg_vals)} mean={np.mean(neg_vals):.4f}"
            )
            sigma_tag = str(sigma).replace(".", "p")
            fname = (
                f"cosim_{args.dataset}_K={args.K_val}{ratio_sfx}"
                f"_cluster{args.cluster}_sigma={sigma_tag}_{augnet_name}{noleg_sfx}.pdf"
            )
            plot_distribution(
                pos_vals, neg_vals,
                args.cluster, sigma,
                os.path.join(args.out_plots, fname),
                args.K_val,
                show_ylabel=(args.show_ylabel and i == 0),
                show_legend=show_leg,
            )


if __name__ == "__main__":
    main()
