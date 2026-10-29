"""Generate a NeurIPS-style LaTeX results table from testMAPs/ files.

Main  : CL1 / CL2 / Avg(CL1+2), table*.
--appendix: 3 tables — CL1+2 detail (no avg) | CL3+4 detail | 4-cluster summary.

Usage
-----
python -B scripts/latex/make_results_table.py [--out attic/results_table.tex]
python -B scripts/latex/make_results_table.py --appendix [--out attic/results_table.tex]
"""

import argparse
import re
from pathlib import Path

DATASETS     = ["ptc_fr", "ptc_fm", "ptc_mr", "cox2"]
CLUSTERS_ALL = [1, 2, 3, 4]
TESTMAPS     = Path("testMAPs")

TASK_PREFIX = (
    "GEN_{ds}_numC=100000_MinR=0.05_MaxR=0.25_"
    "scoring_function=hinge,stemp=0.1,margin=1.0_"
    "score=hinge_esmetric=ap_score_bs=128_"
    "rels-{cl}-{rel}_cs-{suffix}_loss-celoss_nostrat"
)

NOISE_SUFFIX = {
    "sigma=1": "_noise_sigma=1",
}

ROW_ORDER = [
    ("Random",             "random",    None),
    ("k-center",           "kcenter",   None),
    ("k-means",            "kmeans",    None),
    ("GNN dist (L1)",      "gnn_dist",  "all_corpus_negs_K=40_ratio=0.13_dist=l1"),
    ("GNN dist (L2)",      "gnn_dist",  "all_corpus_negs_K=40_ratio=0.13_dist=l2"),
    ("GNN dist (hinge)",   "gnn_dist",  "all_corpus_negs_K=40_ratio=0.13_dist=hinge"),
    ("WL kernel",          "wl_kernel", None),
    ("Infomax",            "infomax",   None),
    ("\\our ($\\sigma=1$)", "lsh",      "sigma=1"),
]

BASELINE_SUFFIXES = {
    "random":    "all_corpus_negs_K=40_ratio=0.13",
    "kcenter":   "all_corpus_negs_K=40_ratio=0.13",
    "kmeans":    "all_corpus_negs_K=40_ratio=0.13",
    "infomax":   "all_corpus_negs_K=40_ratio=0.13_alpha=0.3",
    "wl_kernel": "all_corpus_negs_K=40_ratio=0.13_h=3",
}

AUGNETS = [
    "AugMap_CEMax_Symm_SameModelPerD_Minus_Short_SameLinearPerD",
    "AugMap_CEMax_Symm_SameModelPerD_Minus_Long_SameLinearPerD",
]

DS_NAMES = {"ptc_fr": "PTC-FR", "ptc_fm": "PTC-FM",
            "ptc_mr": "PTC-MR", "cox2":   "COX2"}

LSH_LABEL = "\\our ($\\sigma=1$)"

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


def get_score(rel, ds, cl, extra):
    if rel == "lsh":
        noise_sfx = NOISE_SUFFIX[extra]
        suffix = f"all_corpus_negs_K=40_ratio=0.13{noise_sfx}"
        task_dir = TASK_PREFIX.format(ds=ds, cl=cl, rel="lsh", suffix=suffix)
        candidates = [
            (val, test)
            for aug in AUGNETS
            for val, test in [read_scores(TESTMAPS / task_dir / (aug + ".txt"))]
            if val is not None and test is not None
        ]
        return max(candidates, key=lambda x: x[0])[1] if candidates else None
    else:
        suffix = extra if extra is not None else BASELINE_SUFFIXES.get(rel, "all_corpus_negs_K=40_ratio=0.13")
        task_dir = TASK_PREFIX.format(ds=ds, cl=cl, rel=rel, suffix=suffix)
        return read_map(TESTMAPS / (task_dir + ".txt"))


def build_results():
    """results[label][(ds, cl)] for cl in CLUSTERS_ALL."""
    results = {}
    for label, rel, extra in ROW_ORDER:
        results[label] = {}
        for ds in DATASETS:
            for cl in CLUSTERS_ALL:
                results[label][(ds, cl)] = get_score(rel, ds, cl, extra)
    return results


def fmt(v):
    return "--" if v is None else f"{v:.3f}"


def _row_avg(results, label, ds, cls):
    vals = [results[label][(ds, cl)] for cl in cls if results[label][(ds, cl)] is not None]
    return sum(vals) / len(vals) if len(vals) == len(cls) else None


def _col_rankings(results, ds, col_key):
    """col_key is either a cluster int or a tuple (cls_list,) for avg."""
    if isinstance(col_key, int):
        vals = sorted(
            [results[label][(ds, col_key)] for label, _, _ in ROW_ORDER
             if results[label][(ds, col_key)] is not None],
            reverse=True,
        )
    else:
        cls = col_key
        vals = sorted(
            [v for label, _, _ in ROW_ORDER for v in [_row_avg(results, label, ds, cls)] if v is not None],
            reverse=True,
        )
    best   = vals[0] if vals else None
    second = vals[1] if len(vals) >= 2 else None
    if best is not None and second is not None and abs(second - best) < 1e-6:
        second = vals[2] if len(vals) >= 3 else None
    return best, second


def _apply_bold(cell, v, best, second):
    if v is not None and best is not None and abs(v - best) < 1e-6:
        return rf"\textbf{{{cell}}}"
    if v is not None and second is not None and abs(v - second) < 1e-6:
        return rf"\underline{{{cell}}}"
    return cell


# ---------------------------------------------------------------------------
# ASCII
# ---------------------------------------------------------------------------

def make_ascii(results):
    COLS = [1, 2, (1, 2)]
    rankings = {(ds, col): _col_rankings(results, ds, col if isinstance(col, int) else list(col))
                for ds in DATASETS for col in COLS}

    col_w, label_w = 7, 22
    header1 = f"{'Method':<{label_w}}"
    header2 = f"{'':<{label_w}}"
    for ds in DATASETS:
        span = col_w * 3 + 4 + 2
        header1 += f"  {DS_NAMES[ds]:^{span}}"
        header2 += f"  {'CL1':^{col_w}}  {'CL2':^{col_w}}  {'Avg':^{col_w}}"

    sep = "-" * len(header1)
    lines = ["  * = best  _ = second best", sep, header1, header2, sep]

    for label, rel, extra in ROW_ORDER:
        if label == LSH_LABEL:
            lines.append(sep)
        row = f"{label:<{label_w}}"
        for ds in DATASETS:
            for col in COLS:
                if isinstance(col, int):
                    v = results[label][(ds, col)]
                else:
                    v = _row_avg(results, label, ds, list(col))
                best, second = rankings[(ds, col)]
                cell = fmt(v)
                if v is not None and best is not None and abs(v - best) < 1e-6:
                    cell = f"*{cell}*"
                elif v is not None and second is not None and abs(v - second) < 1e-6:
                    cell = f"_{cell}_"
                row += f"  {cell:^{col_w}}"
        lines.append(row)
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main LaTeX (table*, CL1 / CL2 / Avg(CL1+2))
# ---------------------------------------------------------------------------

def make_latex(results):
    COLS = [1, 2, [1, 2]]
    rankings = {(ds, i): _col_rankings(results, ds, col)
                for ds in DATASETS for i, col in enumerate(COLS)}

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Test MAP at $\budget=40$, $\sigma=1$, $\Qcal_1$ to $\Qcal_2$, all"
        "\n"
        r"  datasets. Results for \our use the best validation score across perturbation"
        "\n"
        r"  training schemes for $\zbphi, \xbphi$. \textbf{Bold}: best per column."
        "\n"
        r"  \underline{Underline}: second best.}"
    )
    lines.append(r"  \label{tab:main_results}")
    lines.append(r"  \resizebox{\columnwidth}{!}{%")
    col_spec = "l" + "".join([" ccc" for _ in DATASETS])
    lines.append(rf"  \begin{{tabular}}{{{col_spec}}}")
    lines.append(r"  \toprule")

    header1 = "  Method"
    for ds in DATASETS:
        header1 += rf" & \multicolumn{{3}}{{c}}{{{DS_NAMES[ds]}}}"
    header1 += r" \\"
    lines.append(header1)

    cmidrules = ""
    for i in range(len(DATASETS)):
        start = 2 + i * 3
        cmidrules += rf"  \cmidrule(lr){{{start}-{start+2}}}"
    lines.append(cmidrules)

    header2 = "  "
    for _ in DATASETS:
        header2 += r" & $\Qcal_1$ & $\Qcal_2$ & Avg"
    header2 += r" \\"
    lines.append(header2)
    lines.append(r"  \midrule")

    for label, rel, extra in ROW_ORDER:
        if label == LSH_LABEL:
            lines.append(r"  \midrule")
        row = f"  {label}"
        for ds in DATASETS:
            for i, col in enumerate(COLS):
                if isinstance(col, int):
                    v = results[label][(ds, col)]
                else:
                    v = _row_avg(results, label, ds, col)
                best, second = rankings[(ds, i)]
                row += " & " + _apply_bold(fmt(v), v, best, second)
        row += r" \\"
        lines.append(row)

    lines.append(r"  \bottomrule")
    lines.append(r"  \end{tabular}%")
    lines.append(r"  }")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Appendix LaTeX (3 tables)
# ---------------------------------------------------------------------------

def _ms(results, label, ds):
    vals = [results[label][(ds, cl)] for cl in CLUSTERS_ALL]
    if any(v is None for v in vals):
        return None, None
    mean = sum(vals) / 4
    std  = (sum((v - mean)**2 for v in vals) / 3) ** 0.5
    return mean, std


def _fmt_ms(mean, std):
    return "--" if mean is None else rf"{mean:.3f} $\pm$ {std:.3f}"


def _col_rankings_mean(results, ds):
    means = sorted(
        [m for label, _, _ in ROW_ORDER for m, _ in [_ms(results, label, ds)] if m is not None],
        reverse=True,
    )
    best   = means[0] if means else None
    second = means[1] if len(means) >= 2 else None
    if best is not None and second is not None and abs(second - best) < 1e-6:
        second = means[2] if len(means) >= 3 else None
    return best, second


def _appendix_dataset_table(results, ds):
    rankings_cl   = {cl: _col_rankings(results, ds, cl) for cl in CLUSTERS_ALL}
    best_m, sec_m = _col_rankings_mean(results, ds)
    lines = []
    lines.append(r"\begin{table*}[h!]")
    lines.append(r"  \centering")
    lines.append(
        rf"  \caption{{Test MAP at $\budget=40$, $\sigma=1$, \textbf{{{DS_NAMES[ds]}}},"
        "\n"
        r"  all four query clusters. Results for \our use the best validation score"
        "\n"
        r"  across perturbation training schemes for $\zbphi, \xbphi$."
        "\n"
        r"  \textbf{Bold}: best per column. \underline{Underline}: second best.}"
    )
    lines.append(rf"  \label{{tab:main_results_{ds}}}")
    col_spec = "l cccc c"
    lines.append(r"  {\small")
    lines.append(r"  \setlength{\tabcolsep}{5pt}")
    lines.append(r"  \renewcommand{\arraystretch}{0.95}")
    lines.append(r"  \begin{adjustbox}{max width=\textwidth}")
    lines.append(rf"  \begin{{tabular}}{{{col_spec}}}")
    lines.append(r"  \toprule")
    lines.append(
        r"  Method & $\Qcal_1$ & $\Qcal_2$ & $\Qcal_3$ & $\Qcal_4$"
        r" & Mean $\pm$ Std \\"
    )
    lines.append(r"  \midrule")
    for label, rel, extra in ROW_ORDER:
        if label == LSH_LABEL:
            lines.append(r"  \midrule")
        row = f"  {label}"
        for cl in CLUSTERS_ALL:
            v = results[label][(ds, cl)]
            best, second = rankings_cl[cl]
            row += " & " + _apply_bold(fmt(v), v, best, second)
        mean, std = _ms(results, label, ds)
        row += " & " + _apply_bold(_fmt_ms(mean, std), mean, best_m, sec_m)
        row += r" \\"
        lines.append(row)
    lines.append(r"  \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \end{adjustbox}")
    lines.append(r"  }")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def make_appendix_latex(results):
    return "\n\n".join(_appendix_dataset_table(results, ds) for ds in DATASETS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", default="attic/eval_cmds_priority/eval_priority.txt")
    parser.add_argument("--out",       default="attic/results_table.tex")
    parser.add_argument("--appendix",  action="store_true")
    args = parser.parse_args()

    results = build_results()

    total = len(ROW_ORDER) * len(DATASETS) * 2  # CL1+2 coverage check
    found = sum(1 for label, _, _ in ROW_ORDER
                for ds in DATASETS for cl in [1, 2]
                if results[label][(ds, cl)] is not None)
    missing = [(label, ds, cl)
               for label, _, _ in ROW_ORDER
               for ds in DATASETS for cl in [1, 2]
               if results[label][(ds, cl)] is None]

    print(f"Coverage (CL1+2): {found}/{total} cells filled")
    if missing:
        print(f"Missing ({len(missing)}):")
        for label, ds, cl in missing:
            print(f"  {label:30s} {ds} CL{cl}")

    ascii_table = make_ascii(results)
    print("\n" + ascii_table)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.appendix:
        tex = make_appendix_latex(results)
        app_path = out_path.with_stem(out_path.stem + "_appendix")
        app_path.write_text(tex + "\n")
        print(f"\nAppendix LaTeX written to {app_path}")
        ascii_out = app_path.with_suffix(".txt")
    else:
        tex = make_latex(results)
        out_path.write_text(tex + "\n")
        print(f"\nLaTeX written to {out_path}")
        ascii_out = out_path.with_suffix(".txt")

    ascii_out.write_text(ascii_table + "\n")
    print(f"ASCII written to {ascii_out}")


if __name__ == "__main__":
    main()
