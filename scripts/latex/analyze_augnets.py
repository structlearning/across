"""Analyze per-AugNet MAP scores for the priority LSH cells.

Scope: CL1+2, K=40, noise variants (no_noise / sigma=0.5 / sigma=1),
       all 4 datasets — exactly the cells in the main results table.

Outputs
-------
1. Per-cell table showing all 4 AugNet scores + which wins
2. Win-count summary per AugNet (how often each is best)
3. Per-AugNet mean MAP (across all available cells)
4. Two LaTeX table snippets: representative AugNet vs best-across-AugNets

Usage
-----
python -B scripts/latex/analyze_augnets.py [--out_dir attic/augnet_analysis]
"""

import argparse
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config (mirrors make_results_table.py)
# ---------------------------------------------------------------------------
DATASETS = ["ptc_fr", "ptc_fm", "ptc_mr", "cox2"]
CLUSTERS = [1, 2]
TESTMAPS = Path("testMAPs")

TASK_PREFIX = (
    "GEN_{ds}_numC=100000_MinR=0.05_MaxR=0.25_"
    "scoring_function=hinge,stemp=0.1,margin=1.0_"
    "score=hinge_esmetric=ap_score_bs=128_"
    "rels-{cl}-{rel}_cs-{suffix}_loss-celoss_nostrat"
)

NOISE_VARIANTS = {
    "no_noise":  "",
    "sigma=0.5": "_noise_sigma=0.5",
    "sigma=1":   "_noise_sigma=1",
}

AUGNETS = [
    "AugMap_CEMax_Symm_SameModelPerD_Minus_Short_SameLinearPerD",
    "AugMap_CEMax_Symm_SameModelPerD_Minus_Long_SameLinearPerD",
    "AugMap_CEAvg_Symm_SameModelPerD_Minus_Short_SameLinearPerD",
    "AugMap_CEAvg_Symm_SameModelPerD_Minus_Long_SameLinearPerD",
]

AUG_SHORT = {
    "AugMap_CEMax_Symm_SameModelPerD_Minus_Short_SameLinearPerD": "CEMax-Short",
    "AugMap_CEMax_Symm_SameModelPerD_Minus_Long_SameLinearPerD":  "CEMax-Long",
    "AugMap_CEAvg_Symm_SameModelPerD_Minus_Short_SameLinearPerD": "CEAvg-Short",
    "AugMap_CEAvg_Symm_SameModelPerD_Minus_Long_SameLinearPerD":  "CEAvg-Long",
}

DS_NAMES = {"ptc_fr": "PTC-FR", "ptc_fm": "PTC-FM",
            "ptc_mr": "PTC-MR", "cox2":   "COX2"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_map(path: Path):
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("test_map_score:"):
            return float(line.split(":")[1].strip())
    return None


def get_aug_scores(ds, cl, noise_key):
    noise_sfx = NOISE_VARIANTS[noise_key]
    suffix = f"all_corpus_negs_K=40_ratio=0.13{noise_sfx}"
    task_dir = TASK_PREFIX.format(ds=ds, cl=cl, rel="lsh", suffix=suffix)
    return {
        aug: read_map(TESTMAPS / task_dir / (aug + ".txt"))
        for aug in AUGNETS
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze():
    # cell_data[(ds, cl, noise)] = {aug: score_or_None}
    cell_data = {}
    for ds in DATASETS:
        for cl in CLUSTERS:
            for noise_key in NOISE_VARIANTS:
                cell_data[(ds, cl, noise_key)] = get_aug_scores(ds, cl, noise_key)

    # --- Win counts and mean per AugNet ---
    win_counts = defaultdict(int)    # aug -> # cells where it's best
    tie_counts = defaultdict(int)    # aug -> # ties for best
    scores_all = defaultdict(list)   # aug -> [scores]

    cells_with_data = []
    for key, aug_scores in cell_data.items():
        available = {a: v for a, v in aug_scores.items() if v is not None}
        if not available:
            continue
        cells_with_data.append(key)
        best_val = max(available.values())
        winners = [a for a, v in available.items() if abs(v - best_val) < 1e-6]
        for a in winners:
            if len(winners) == 1:
                win_counts[a] += 1
            else:
                tie_counts[a] += 1
        for a, v in available.items():
            scores_all[a].append(v)

    # --- Representative AugNet: most wins (ties broken by mean MAP) ---
    def aug_rank_key(aug):
        wins = win_counts[aug]
        ties = tie_counts[aug]
        mean = sum(scores_all[aug]) / len(scores_all[aug]) if scores_all[aug] else 0
        return (wins + ties * 0.5, mean)

    sorted_augs = sorted(AUGNETS, key=aug_rank_key, reverse=True)
    representative = sorted_augs[0]

    return cell_data, win_counts, tie_counts, scores_all, cells_with_data, representative


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_per_cell_table(cell_data, win_counts):
    short = AUG_SHORT
    aug_cols = [short[a] for a in AUGNETS]
    col_w = 11
    label_w = 32

    header = f"{'Cell':<{label_w}}" + "".join(f"  {c:^{col_w}}" for c in aug_cols) + f"  {'Winner':^14}"
    sep = "-" * len(header)
    print("\n=== Per-cell AugNet scores (K=40) ===")
    print(sep)
    print(header)
    print(sep)

    prev_noise = None
    for ds in DATASETS:
        for noise_key in NOISE_VARIANTS:
            for cl in CLUSTERS:
                key = (ds, cl, noise_key)
                aug_scores = cell_data[key]
                available = {a: v for a, v in aug_scores.items() if v is not None}
                if not available:
                    continue
                best_val = max(available.values())
                winners = [short[a] for a, v in available.items() if abs(v - best_val) < 1e-6]

                label = f"{DS_NAMES[ds]} CL{cl} {noise_key}"
                row = f"{label:<{label_w}}"
                for aug in AUGNETS:
                    v = aug_scores[aug]
                    if v is None:
                        cell = "--"
                    elif abs(v - best_val) < 1e-6:
                        cell = f"*{v:.3f}*"
                    else:
                        cell = f"{v:.3f}"
                    row += f"  {cell:^{col_w}}"
                row += f"  {', '.join(winners)}"
                print(row)

    print(sep)


def print_summary(win_counts, tie_counts, scores_all, representative):
    print("\n=== AugNet summary ===")
    print(f"{'AugNet':<14}  {'Wins':>5}  {'Ties':>5}  {'N':>4}  {'Mean MAP':>9}  {'Min':>7}  {'Max':>7}")
    print("-" * 60)
    sorted_augs = sorted(AUGNETS, key=lambda a: (
        win_counts[a] + tie_counts[a] * 0.5,
        sum(scores_all[a]) / len(scores_all[a]) if scores_all[a] else 0
    ), reverse=True)
    for aug in sorted_augs:
        name = AUG_SHORT[aug]
        vals = scores_all[aug]
        mean = sum(vals) / len(vals) if vals else float("nan")
        mn   = min(vals) if vals else float("nan")
        mx   = max(vals) if vals else float("nan")
        marker = " <-- representative" if aug == representative else ""
        print(f"{name:<14}  {win_counts[aug]:>5}  {tie_counts[aug]:>5}  {len(vals):>4}  {mean:>9.4f}  {mn:>7.3f}  {mx:>7.3f}{marker}")

    print(f"\nRepresentative AugNet: {AUG_SHORT[representative]}")
    print(f"  ({representative})")


def print_augnet_comparison_table(cell_data):
    """4 blocks (one per AugNet), 3 rows each (noise variant), columns = dataset×cluster."""
    noise_labels = {
        "no_noise":  "no noise",
        "sigma=0.5": "sigma=0.5",
        "sigma=1":   "sigma=1",
    }

    col_w   = 7
    label_w = 12

    # Header
    header1 = f"{'':^{label_w}}"
    header2 = f"{'':^{label_w}}"
    for ds in DATASETS:
        span = col_w * 2 + 3
        header1 += f"  {DS_NAMES[ds]:^{span}}"
        header2 += f"  {'CL1':^{col_w}} {'CL2':^{col_w}}"

    thin  = "-" * len(header1)
    thick = "=" * len(header1)
    print("\n=== AugNet comparison table (K=40, CL1+2) ===")
    print("  * = best noise variant for that dataset/cluster within this AugNet")

    for aug in AUGNETS:
        # Within-block best per (ds, cl)
        col_best = {}
        for ds in DATASETS:
            for cl in CLUSTERS:
                vals = {nk: cell_data[(ds, cl, nk)][aug] for nk in NOISE_VARIANTS}
                available = {nk: v for nk, v in vals.items() if v is not None}
                col_best[(ds, cl)] = max(available.values()) if available else None

        print(thick)
        print(f"  {AUG_SHORT[aug]}")
        print(thick)
        print(header1)
        print(header2)
        print(thin)
        for noise_key, nlabel in noise_labels.items():
            row = f"{nlabel:<{label_w}}"
            for ds in DATASETS:
                for cl in CLUSTERS:
                    v = cell_data[(ds, cl, noise_key)][aug]
                    best = col_best[(ds, cl)]
                    if v is None:
                        cell = "--"
                    elif best is not None and abs(v - best) < 1e-6:
                        cell = f"*{v:.3f}*"
                    else:
                        cell = f"{v:.3f}"
                    row += f"  {cell:^{col_w}}"
            print(row)
    print(thick)


def print_consistency_by_dataset(cell_data):
    """For each dataset, which AugNet wins most across CL1/2 and noise variants."""
    print("\n=== Per-dataset AugNet win counts ===")
    print(f"{'Dataset':<8}  " + "  ".join(f"{AUG_SHORT[a]:>14}" for a in AUGNETS))
    print("-" * 75)
    for ds in DATASETS:
        ds_wins = defaultdict(int)
        for cl in CLUSTERS:
            for noise_key in NOISE_VARIANTS:
                aug_scores = cell_data[(ds, cl, noise_key)]
                available = {a: v for a, v in aug_scores.items() if v is not None}
                if not available:
                    continue
                best_val = max(available.values())
                for a, v in available.items():
                    if abs(v - best_val) < 1e-6:
                        ds_wins[a] += 1
        print(f"{DS_NAMES[ds]:<8}  " + "  ".join(f"{ds_wins[a]:>14}" for a in AUGNETS))


# ---------------------------------------------------------------------------
# LaTeX snippets
# ---------------------------------------------------------------------------

def make_lsh_latex_rows(cell_data, representative):
    """Return (repr_rows, best_rows) as lists of LaTeX row strings."""
    noise_labels = {
        "no_noise":  r"LSH (no noise)",
        "sigma=0.5": r"LSH ($\sigma=0.5$)",
        "sigma=1":   r"LSH ($\sigma=1$)",
    }

    def fmt(v):
        return "--" if v is None else f"{v:.3f}"

    def col_max(noise_key):
        vals = {}
        for ds in DATASETS:
            for cl in CLUSTERS:
                aug_scores = cell_data[(ds, cl, noise_key)]
                v_repr = aug_scores[representative]
                v_best = max((x for x in aug_scores.values() if x is not None), default=None)
                vals[(ds, cl, "repr")] = v_repr
                vals[(ds, cl, "best")] = v_best
        return vals

    repr_rows, best_rows = [], []
    for noise_key, label in noise_labels.items():
        vals = col_max(noise_key)
        # per-col max for bolding
        repr_bests = {(ds, cl): None for ds in DATASETS for cl in CLUSTERS}
        best_bests = {(ds, cl): None for ds in DATASETS for cl in CLUSTERS}
        for ds in DATASETS:
            for cl in CLUSTERS:
                repr_bests[(ds, cl)] = vals[(ds, cl, "repr")]
                best_bests[(ds, cl)] = vals[(ds, cl, "best")]

        def make_row(cell_vals, row_label):
            row = f"  {row_label}"
            for ds in DATASETS:
                for cl in CLUSTERS:
                    v = cell_vals[(ds, cl)]
                    row += f" & {fmt(v)}"
            row += r" \\"
            return row

        repr_rows.append(make_row(repr_bests, label))
        best_rows.append(make_row(best_bests, label))

    return repr_rows, best_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import io, sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="attic/augnet_analysis")
    args = parser.parse_args()

    cell_data, win_counts, tie_counts, scores_all, cells_with_data, representative = analyze()

    total_cells = len(DATASETS) * len(CLUSTERS) * len(NOISE_VARIANTS)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "augnet_analysis.txt"

    buf = io.StringIO()
    tee = sys.stdout

    class Tee:
        def write(self, s):
            tee.write(s)
            buf.write(s)
        def flush(self):
            tee.flush()

    sys.stdout = Tee()

    print(f"Coverage: {len(cells_with_data)}/{total_cells} LSH cells have at least one AugNet result")
    print_augnet_comparison_table(cell_data)
    print_per_cell_table(cell_data, win_counts)
    print_summary(win_counts, tie_counts, scores_all, representative)
    print_consistency_by_dataset(cell_data)

    repr_rows, best_rows = make_lsh_latex_rows(cell_data, representative)
    print(f"\n=== LaTeX LSH rows (representative: {AUG_SHORT[representative]}) ===")
    for r in repr_rows:
        print(r)
    print(f"\n=== LaTeX LSH rows (best across AugNets) ===")
    for r in best_rows:
        print(r)

    sys.stdout = tee
    out_path.write_text(buf.getvalue())
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
