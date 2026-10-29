"""Generate ablation table comparing \our variants (LaTeX + ASCII).

Scope: K=40, ratio=0.13, sigma=1, all 4 datasets.
Main  : avg MAP over CL1+2, wraptable.
--appendix: 3 tables — CL1+2 detail | CL3+4 detail | 4-cluster summary.

Usage
-----
python -B scripts/latex/make_ablation_table.py [--out attic/ablation_table.tex]
python -B scripts/latex/make_ablation_table.py --appendix [--out attic/ablation_table.tex]
"""

import argparse
import io
import sys
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

SUFFIX = "all_corpus_negs_K=40_ratio=0.13_noise_sigma=1"

DS_NAMES = {"ptc_fr": "PTC-FR", "ptc_fm": "PTC-FM",
            "ptc_mr": "PTC-MR", "cox2":   "COX2"}

BASE = "AugMap_CEMax_Symm_SameModelPerD_Minus_{form}_SameLinearPerD"

ROWS = [
    (r"\our (rand)", [
        BASE.format(form="Long") + "_random_init",
    ]),
    (r"\our (aug)", [
        BASE.format(form="Short") + "_pre_augnet_replicas",
        BASE.format(form="Long")  + "_pre_augnet_replicas",
    ]),
    (r"\our", [
        BASE.format(form="Short"),
        BASE.format(form="Long"),
    ]),
]

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


def get_best(augnets, ds, cl):
    task_dir = TASK_PREFIX.format(ds=ds, cl=cl, rel="lsh", suffix=SUFFIX)
    candidates = [
        (val, test)
        for aug in augnets
        for val, test in [read_scores(TESTMAPS / task_dir / (aug + ".txt"))]
        if val is not None and test is not None
    ]
    return max(candidates, key=lambda x: x[0])[1] if candidates else None


def build_data():
    """data[(row_label, ds, cl)] for cl in CLUSTERS_ALL."""
    data = {}
    for row_label, augnets in ROWS:
        for ds in DATASETS:
            for cl in CLUSTERS_ALL:
                data[(row_label, ds, cl)] = get_best(augnets, ds, cl)
    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt(v):
    return "--" if v is None else f"{v:.3f}"


def _row_avg(data, row_label, ds, cls):
    vals = [data[(row_label, ds, cl)] for cl in cls if data[(row_label, ds, cl)] is not None]
    return sum(vals) / len(vals) if len(vals) == len(cls) else None


def _col_rankings_avg(data, ds, cls):
    vals = sorted(
        [v for rl, _ in ROWS for v in [_row_avg(data, rl, ds, cls)] if v is not None],
        reverse=True,
    )
    best   = vals[0] if vals else None
    second = vals[1] if len(vals) >= 2 else None
    if best is not None and second is not None and abs(second - best) < 1e-6:
        second = vals[2] if len(vals) >= 3 else None
    return best, second


def _col_rankings_cl(data, ds, cl):
    vals = sorted(
        [data[(rl, ds, cl)] for rl, _ in ROWS if data[(rl, ds, cl)] is not None],
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

def make_ascii(data):
    col_w, label_w = 7, 24
    rankings = {ds: _col_rankings_avg(data, ds, [1, 2]) for ds in DATASETS}
    header = f"{'':^{label_w}}" + "".join(f"  {DS_NAMES[ds]:^{col_w}}" for ds in DATASETS)
    sep = "-" * len(header)
    lines = ["  * = best  _ = second best", sep, header, sep]
    for row_label, _ in ROWS:
        row = f"{row_label:<{label_w}}"
        for ds in DATASETS:
            v = _row_avg(data, row_label, ds, [1, 2])
            best, second = rankings[ds]
            cell = "--" if v is None else f"{v:.3f}"
            if v is not None and best is not None and abs(v - best) < 1e-6:
                cell = f"*{cell}*"
            elif v is not None and second is not None and abs(v - second) < 1e-6:
                cell = f"_{cell}_"
            row += f"  {cell:^{col_w}}"
        lines.append(row)
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main LaTeX (wraptable, avg of CL1+2)
# ---------------------------------------------------------------------------

def make_latex(data):
    rankings = {ds: _col_rankings_avg(data, ds, [1, 2]) for ds in DATASETS}

    lines = []
    lines.append(r"\begin{wraptable}{r}{0.50\textwidth}")
    lines.append(r"  \centering")
    lines.append(r"  \vspace{-1.5ex}")
    lines.append(
        r"  \caption{Ablation of trained components. Test MAP at $\budget = 40$, $\sigma = 1$,"
        "\n"
        r"  $\Qcal_1$ to $\Qcal_2$, all datasets. Each cell: best MAP across training"
        "\n"
        r"  configurations, averaged over $\Qcal_1$ and $\Qcal_2$, selected by validation score."
        "\n"
        r"  \textbf{Bold}: best per column. \underline{Underline}: second best.}"
    )
    lines.append(r"  \label{tab:ablation}")
    col_spec = "l" + "".join(" c" for _ in DATASETS)
    lines.append(r"  \resizebox{\linewidth}{!}{%")
    lines.append(rf"  \begin{{tabular}}{{{col_spec}}}")
    lines.append(r"  \toprule")
    h1 = "  Method"
    for ds in DATASETS:
        h1 += rf" & {DS_NAMES[ds]}"
    h1 += r" \\"
    lines.append(h1)
    lines.append(r"  \midrule")
    for row_label, _ in ROWS:
        row = f"  {row_label}"
        for ds in DATASETS:
            v = _row_avg(data, row_label, ds, [1, 2])
            best, second = rankings[ds]
            row += " & " + _apply_bold(fmt(v), v, best, second)
        row += r" \\"
        lines.append(row)
    lines.append(r"  \bottomrule")
    lines.append(r"  \end{tabular}%")
    lines.append(r"  }")
    lines.append(r"  \vspace{-1ex}")
    lines.append(r"\end{wraptable}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Appendix LaTeX (3 tables)
# ---------------------------------------------------------------------------

_ABLATION_CAPTION_BODY = (
    r"  \our (rand): $\nsrbase$ and $\zbphi$, $\xbphi$ randomly initialised, no training."
    "\n"
    r"  \our (aug): trained $\nsrbase$; $f_\phi$ replaced by the $R$-replica randomised"
    "\n"
    r"  feature map. \our: full method."
    "\n"
    r"  Each cell: best MAP across training configurations, selected by validation score."
    "\n"
    r"  \textbf{Bold}: best per column. \underline{Underline}: second best.}"
)


def _ms(data, row_label, ds):
    vals = [data[(row_label, ds, cl)] for cl in CLUSTERS_ALL]
    if any(v is None for v in vals):
        return None, None
    mean = sum(vals) / 4
    std  = (sum((v - mean)**2 for v in vals) / 3) ** 0.5
    return mean, std


def _fmt_ms(mean, std):
    return "--" if mean is None else rf"{mean:.3f} $\pm$ {std:.3f}"


def _col_rankings_mean(data, ds):
    means = sorted(
        [m for rl, _ in ROWS for m, _ in [_ms(data, rl, ds)] if m is not None],
        reverse=True,
    )
    best   = means[0] if means else None
    second = means[1] if len(means) >= 2 else None
    if best is not None and second is not None and abs(second - best) < 1e-6:
        second = means[2] if len(means) >= 3 else None
    return best, second


def _appendix_dataset_table(data, ds):
    rankings_cl   = {cl: _col_rankings_cl(data, ds, cl) for cl in CLUSTERS_ALL}
    best_m, sec_m = _col_rankings_mean(data, ds)
    lines = []
    lines.append(r"\begin{table*}[h!]")
    lines.append(r"  \centering")
    lines.append(
        rf"  \caption{{Ablation of trained components. Test MAP at $\budget=40$, $\sigma=1$,"
        "\n"
        rf"  \textbf{{{DS_NAMES[ds]}}}, all four query clusters."
        "\n" + _ABLATION_CAPTION_BODY
    )
    lines.append(rf"  \label{{tab:ablation_{ds}}}")
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
    for row_label, _ in ROWS:
        row = f"  {row_label}"
        for cl in CLUSTERS_ALL:
            v = data[(row_label, ds, cl)]
            best, second = rankings_cl[cl]
            row += " & " + _apply_bold(fmt(v), v, best, second)
        mean, std = _ms(data, row_label, ds)
        row += " & " + _apply_bold(_fmt_ms(mean, std), mean, best_m, sec_m)
        row += r" \\"
        lines.append(row)
    lines.append(r"  \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \end{adjustbox}")
    lines.append(r"  }")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def make_appendix_latex(data):
    return "\n\n".join(_appendix_dataset_table(data, ds) for ds in DATASETS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",      default="attic/ablation_table.tex")
    parser.add_argument("--appendix", action="store_true")
    args = parser.parse_args()

    data = build_data()

    buf = io.StringIO()
    tee = sys.stdout

    class Tee:
        def write(self, s): tee.write(s); buf.write(s)
        def flush(self): tee.flush()

    sys.stdout = Tee()
    print(make_ascii(data))
    sys.stdout = tee

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.appendix:
        tex = make_appendix_latex(data)
        app_path = out_path.with_stem(out_path.stem + "_appendix")
        app_path.write_text(tex + "\n")
        print(f"\nAppendix LaTeX written to {app_path}")
        ascii_out = app_path.with_suffix(".txt")
    else:
        tex = make_latex(data)
        out_path.write_text(tex + "\n")
        print(f"\nLaTeX written to {out_path}")
        ascii_out = out_path.with_suffix(".txt")

    ascii_out.write_text(buf.getvalue())
    print(f"ASCII written to {ascii_out}")


if __name__ == "__main__":
    main()
