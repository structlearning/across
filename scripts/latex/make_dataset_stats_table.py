"""Generate dataset statistics table (LaTeX + ASCII).

One table per dataset. Rows: train Q1-Q4, val (shared), test Q1-Q4.
Columns: #Queries, Avg nodes, Avg edges, Avg pos, Pos rate (%).
Corpus size (100K) reported in caption.

Usage
-----
python -B scripts/latex/make_dataset_stats_table.py [--out attic/dataset_stats_table.tex]
"""

import argparse
import json
import pickle
from pathlib import Path

DATASETS    = ["ptc_fr", "ptc_fm", "ptc_mr", "cox2"]
CLUSTERS    = [1, 2, 3, 4]
CORPUS_SIZE = 100_000
DATA_ROOT   = Path("data")

DS_NAMES = {"ptc_fr": "PTC-FR", "ptc_fm": "PTC-FM",
            "ptc_mr": "PTC-MR", "cox2":   "COX2"}

QUERIES_FNAME = "{split}_queries_for_100000_corpus_subgraphs_min15_max20_relabeled_range_0.05_to_0.25.pkl"
REL_FNAME     = "{split}_rel_dict_for_100000_corpus_subgraphs_min15_max20_range_0.05_to_0.25.pkl"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_queries(ds, split):
    path = DATA_ROOT / ds / "preprocessed" / "splits" / split / QUERIES_FNAME.format(split=split)
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_rel(ds, split):
    path = DATA_ROOT / ds / "preprocessed" / "splits" / split / REL_FNAME.format(split=split)
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_ca(ds, split):
    path = DATA_ROOT / ds / "preprocessed" / "splits" / split / "cluster_assignments.json"
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _graph_stats(graphs):
    nodes = [g.number_of_nodes() for g in graphs]
    edges = [g.number_of_edges() for g in graphs]
    n = len(graphs)
    return (
        sum(nodes) / n, min(nodes), max(nodes),
        sum(edges) / n, min(edges), max(edges),
    )


def _pos_avg(rel, indices):
    counts = [len(rel[q]["pos"]) for q in indices if q in rel]
    return sum(counts) / len(counts) if counts else 0.0


# ---------------------------------------------------------------------------
# Build data
# ---------------------------------------------------------------------------

def build_data():
    """data[(ds, split, cl)] = (n_queries, avg_n, min_n, max_n, avg_e, min_e, max_e, avg_pos, pos_rate)
    cl is 1-4 for train/test, None for val."""
    data = {}
    for ds in DATASETS:
        for split in ["train", "test"]:
            graphs = _load_queries(ds, split)
            ca     = _load_ca(ds, split)
            rel    = _load_rel(ds, split)
            for cl in CLUSTERS:
                idxs  = [i for i, c in enumerate(ca) if c == cl]
                stats = _graph_stats([graphs[i] for i in idxs])
                avg_p = _pos_avg(rel, idxs)
                data[(ds, split, cl)] = (len(idxs), *stats, avg_p, avg_p / CORPUS_SIZE * 100)

        graphs  = _load_queries(ds, "val")
        rel     = _load_rel(ds, "val")
        stats   = _graph_stats(graphs)
        counts  = [len(v["pos"]) for v in rel.values()]
        avg_p   = sum(counts) / len(counts)
        data[(ds, "val", None)] = (len(graphs), *stats, avg_p, avg_p / CORPUS_SIZE * 100)

    return data


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_n(n):    return str(n)
def _fmt_f(v):    return f"{v:.1f}"
def _fmt_avg(v):  return f"{v:.0f}"
def _fmt_rate(v): return f"{v:.1f}"


# ---------------------------------------------------------------------------
# ASCII
# ---------------------------------------------------------------------------

def make_ascii(data):
    lines = []
    header = f"{'':20s}  {'#Q':>4}  {'AvgN':>5}  {'MinN':>4}  {'MaxN':>4}  {'AvgE':>5}  {'MinE':>4}  {'MaxE':>4}  {'Avg pos':>8}  {'Pos%':>5}"
    sep = "-" * len(header)
    lines.append(header)
    lines.append(sep)
    for ds in DATASETS:
        lines.append(DS_NAMES[ds])
        for cl in CLUSTERS:
            n, an, minn, maxn, ae, mine, maxe, ap, rate = data[(ds, "train", cl)]
            lines.append(f"  Train Q{cl}            {n:>4}  {an:>5.1f}  {minn:>4}  {maxn:>4}  {ae:>5.1f}  {mine:>4}  {maxe:>4}  {ap:>8.0f}  {rate:>4.1f}%")
        n, an, minn, maxn, ae, mine, maxe, ap, rate = data[(ds, "val", None)]
        lines.append(f"  Val (shared)        {n:>4}  {an:>5.1f}  {minn:>4}  {maxn:>4}  {ae:>5.1f}  {mine:>4}  {maxe:>4}  {ap:>8.0f}  {rate:>4.1f}%")
        for cl in CLUSTERS:
            n, an, minn, maxn, ae, mine, maxe, ap, rate = data[(ds, "test", cl)]
            lines.append(f"  Test Q{cl}             {n:>4}  {an:>5.1f}  {minn:>4}  {maxn:>4}  {ae:>5.1f}  {mine:>4}  {maxe:>4}  {ap:>8.0f}  {rate:>4.1f}%")
        lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

def _dataset_table(data, ds):
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"  \centering")
    lines.append(
        rf"  \caption{{Query statistics for \textbf{{{DS_NAMES[ds]}}}."
        "\n"
        r"  Corpus: 100\,000 graphs. N: nodes per query graph; E: edges per query graph."
        "\n"
        r"  Pos rate = avg positives / corpus size. Val is shared across all clusters.}"
    )
    lines.append(rf"  \label{{tab:dataset_stats_{ds}}}")
    lines.append(r"  \begin{tabular}{l r r r r r r r r r}")
    lines.append(r"  \toprule")
    lines.append(r"  Split & \#Q & Avg\,N & Min\,N & Max\,N & Avg\,E & Min\,E & Max\,E & Avg pos & Pos rate (\%) \\")
    lines.append(r"  \midrule")

    lines.append(r"  \multicolumn{10}{l}{\textit{Train}} \\")
    for cl in CLUSTERS:
        n, an, minn, maxn, ae, mine, maxe, ap, rate = data[(ds, "train", cl)]
        lines.append(
            rf"  \quad $\Qcal_{cl}$ & {_fmt_n(n)} & {_fmt_f(an)} & {minn} & {maxn}"
            rf" & {_fmt_f(ae)} & {mine} & {maxe} & {_fmt_avg(ap)} & {_fmt_rate(rate)} \\"
        )

    lines.append(r"  \midrule")
    n, an, minn, maxn, ae, mine, maxe, ap, rate = data[(ds, "val", None)]
    lines.append(
        rf"  Val (shared) & {_fmt_n(n)} & {_fmt_f(an)} & {minn} & {maxn}"
        rf" & {_fmt_f(ae)} & {mine} & {maxe} & {_fmt_avg(ap)} & {_fmt_rate(rate)} \\"
    )

    lines.append(r"  \midrule")
    lines.append(r"  \multicolumn{10}{l}{\textit{Test}} \\")
    for cl in CLUSTERS:
        n, an, minn, maxn, ae, mine, maxe, ap, rate = data[(ds, "test", cl)]
        lines.append(
            rf"  \quad $\Qcal_{cl}$ & {_fmt_n(n)} & {_fmt_f(an)} & {minn} & {maxn}"
            rf" & {_fmt_f(ae)} & {mine} & {maxe} & {_fmt_avg(ap)} & {_fmt_rate(rate)} \\"
        )

    lines.append(r"  \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def make_latex(data):
    return "\n\n".join(_dataset_table(data, ds) for ds in DATASETS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="attic/dataset_stats_table.tex")
    args = parser.parse_args()

    data = build_data()

    ascii_table = make_ascii(data)
    print(ascii_table)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(make_latex(data) + "\n")
    print(f"\nLaTeX written to {out_path}")

    ascii_path = out_path.with_suffix(".txt")
    ascii_path.write_text(ascii_table + "\n")
    print(f"ASCII written to {ascii_path}")


if __name__ == "__main__":
    main()
