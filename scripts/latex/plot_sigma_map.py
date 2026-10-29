"""Plot test MAP vs noise sigma for \our at K=40, single cluster.

One figure per dataset, single curve (sigma on log-scale X, test MAP on Y).
AugNet selection: best val_ap_score across all 4 AugNets.

Usage
-----
python -B scripts/latex/plot_sigma_map.py [--cluster 1] [--out-dir plots/sigma_map]
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASETS = ["ptc_fr", "ptc_fm", "ptc_mr", "cox2"]
DS_NAMES = {"ptc_fr": "PTC-FR", "ptc_fm": "PTC-FM",
            "ptc_mr": "PTC-MR", "cox2":   "COX2"}
TESTMAPS = Path("testMAPs")

TASK_PREFIX = (
    "GEN_{ds}_numC=100000_MinR=0.05_MaxR=0.25_"
    "scoring_function=hinge,stemp=0.1,margin=1.0_"
    "score=hinge_esmetric=ap_score_bs=128_"
    "rels-{cl}-lsh_cs-all_corpus_negs_K=40_ratio=0.13_noise_sigma={sigma}_loss-celoss_nostrat"
)

# WARNING: This pool selects across TWO axes at once — training scheme
# {CEMax (adversarial), CEAvg (sampling)} AND augmentation form {Short, Long}.
# The main results (make_results_table.py, Tables 6-9) select over FORMS only,
# within adversarial (CEMax) training. So the sigma-MAP panels here are NOT
# directly reconcilable cell-for-cell with the headline tables: a given
# (dataset, cluster, sigma) point may be won by a CEAvg net that never appears
# in the main tables. Restrict to the two CEMax nets to match the headline
# selection before treating any value here as equal to the corresponding table
# cell. Left as-is intentionally (density shapes are near-identical); see the
# short/long two-axis note in make_short_long_table.py.
AUGNETS = [
    "AugMap_CEMax_Symm_SameModelPerD_Minus_Short_SameLinearPerD",
    "AugMap_CEMax_Symm_SameModelPerD_Minus_Long_SameLinearPerD",
    "AugMap_CEAvg_Symm_SameModelPerD_Minus_Short_SameLinearPerD",
    "AugMap_CEAvg_Symm_SameModelPerD_Minus_Long_SameLinearPerD",
]

SIGMAS = [0.1, 0.5, 1, 10]

STYLE = {
    "text.usetex":         True,
    "text.latex.preamble": r"\boldmath",
    "font.family":         "serif",
    "font.size":           11,
    "axes.labelsize":      21,
    "axes.titlesize":      20,
    "xtick.labelsize":     18,
    "ytick.labelsize":     18,
    "figure.dpi":          150,
}

DS_COLOR = "#1558a0"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_scores(path: Path):
    if not path.exists():
        return None, None
    val, test = None, None
    for line in path.read_text().splitlines():
        if line.startswith("val_ap_score:"):
            val = float(line.split(":")[1].strip())
        elif line.startswith("test_map_score:"):
            test = float(line.split(":")[1].strip())
    return val, test


def get_best_test_map(ds, cl, sigma):
    task_dir = TASK_PREFIX.format(ds=ds, cl=cl, sigma=sigma)
    candidates = [
        (val, test)
        for aug in AUGNETS
        for val, test in [read_scores(TESTMAPS / task_dir / (aug + ".txt"))]
        if val is not None and test is not None
    ]
    return max(candidates, key=lambda x: x[0])[1] if candidates else None


def build_data(cluster):
    """data[(ds, sigma)] = test MAP for the given cluster, or None if missing."""
    data = {}
    for ds in DATASETS:
        for sigma in SIGMAS:
            data[(ds, sigma)] = get_best_test_map(ds, cluster, sigma)
    return data


# ---------------------------------------------------------------------------
# Plot — one per dataset
# ---------------------------------------------------------------------------

def make_dataset_plot(data, ds, out_path):
    xs = [s for s in SIGMAS if data[(ds, s)] is not None]
    ys = [data[(ds, s)] for s in xs]

    if not xs:
        print(f"  No data for {DS_NAMES[ds]}, skipping.")
        return

    for x, y in zip(xs, ys):
        print(f"  {DS_NAMES[ds]:7s}  sigma={x:<4}  MAP={y:.4f}")

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(3.5, 2.8))

        xs_pos = list(range(len(xs)))
        sigma_labels = [f"${s}$" for s in xs]
        ax.plot(xs_pos, ys, color=DS_COLOR, marker="o", linewidth=1.8, markersize=6)

        ax.set_xticks(xs_pos)
        ax.set_xticklabels(sigma_labels)
        ax.set_xlabel(r"\textbf{Noise} $\sigma$ $\rightarrow$")
        ax.set_ylabel(r"\textbf{Test MAP} $\rightarrow$")
        ax.grid(True, which="both", linestyle="--", linewidth=1.0, alpha=0.75)

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=int, default=1,
                        help="Cluster to plot (default: 1)")
    parser.add_argument("--out-dir", default="plots/sigma_map")
    parser.add_argument("--appendix", action="store_true",
                        help="Save to plots/appendix/sigma_map/ instead of plots/sigma_map/")
    args = parser.parse_args()

    if args.appendix and args.out_dir == parser.get_default("out_dir"):
        args.out_dir = os.path.join("plots", "appendix", "sigma_map")

    data = build_data(args.cluster)

    missing = [(ds, s) for ds in DATASETS for s in SIGMAS if data[(ds, s)] is None]
    if missing:
        print(f"Missing {len(missing)} (ds, sigma) pairs:")
        for ds, s in missing:
            print(f"  {DS_NAMES[ds]}  sigma={s}")
    print()

    os.makedirs(args.out_dir, exist_ok=True)
    for ds in DATASETS:
        out_path = os.path.join(args.out_dir, f"sigma_map_{ds}_cl{args.cluster}.pdf")
        make_dataset_plot(data, ds, out_path)


if __name__ == "__main__":
    main()
