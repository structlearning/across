"""Common plotting utilities for LSH retrieval sweep results."""

import importlib.util
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from reform.lsh.constants import TOP_KS

# ---------------------------------------------------------------------------
# Import convert_name from scripts/misc/wandb_to_human.py
# ---------------------------------------------------------------------------

def _load_convert_name():
    script = Path(__file__).parents[2] / "scripts" / "misc" / "wandb_to_human.py"
    spec = importlib.util.spec_from_file_location("wandb_to_human", script)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.convert_name


_convert_name = _load_convert_name()


# ---------------------------------------------------------------------------
# Metric list
# ---------------------------------------------------------------------------

# Deduplicated K values in stable order
_UNIQUE_TOP_KS = list(dict.fromkeys(TOP_KS))

# (result_key, display_label) pairs for every metric
_METRICS: list[tuple[str, str]] = [
    ("map", "MAP"),
    ("map@50", "MAP@50"),
    ("set_prec", "Set Precision"),
    ("set_rec", "Set Recall"),
    ("mrr", "MRR"),
    *[(f"prec@{k}", f"Prec@{k}") for k in _UNIQUE_TOP_KS],
    *[(f"rec@{k}", f"Rec@{k}") for k in _UNIQUE_TOP_KS],
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_corpus_size(
    all_results: dict[tuple[float | None, bool | None, str], list[dict] | None],
) -> int | None:
    """Infer corpus size from subset_size=0 entries (exhaustive LSH retrieval)."""
    for (_, _, wtype), res in all_results.items():
        if res is None or wtype != "LSH":
            continue
        for entry in res:
            if entry["subset_size"] == 0:
                return int(round(entry["num_evals"]))
    return None


def _aug_readable_name(task_name: str) -> str:
    """Convert a full task name to a human-readable aug name for the plot title."""
    aug_task_name = task_name.split("_lsh_")[0]
    return _convert_name(aug_task_name)


def make_plot_name(dataset_name: str, task_name: str, rerank_mode: str) -> str:
    """Return a human-readable plot name from dataset + full task name."""
    aug_readable_name = _aug_readable_name(task_name)
    return f"{dataset_name}_{aug_readable_name}_{rerank_mode}"


def _write_run_info(
    out_dir: Path,
    task_name: str,
    base_overrides: list[str],
    pkl_name: str,
    sweep_script: str = "python -m reform.runner.lsh_retr_sweep",
) -> None:
    """Write a txt file with the full checkpoint name and ready-to-run commands."""
    overrides_str = " ".join(base_overrides)
    lines = [
        f"Checkpoint : {task_name}",
        "",
        "# Sweep command (re-run the full retrieval sweep):",
        f"{sweep_script} {overrides_str}",
        "",
        "# Plot command (regenerate plots from saved pickle):",
        f"python -m reform.lsh_retr_plot {pkl_name} {overrides_str}",
    ]
    (out_dir / "run_info.txt").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_sweep_results(
    all_results: dict[tuple[float | None, bool | None, str], list[dict] | None],
    base_dir: str | Path,
    dataset_name: str,
    task_name: str,
    base_overrides: list[str],
    corpus_size: int | None = None,
    inner_dir: str | None = None,
    random_hyper_seed: int | None = None,
    trained_loss_type: str | None = None,
    noise_sigma_query: float = 0.0,
    noise_sigma_corpus: float = 0.0,
    rerank_mode: str = "cos",
    sweep_script: str = "python -m reform.runner.lsh_retr_sweep",
) -> Path:
    """Generate and save scatterplots for every metric in *all_results*.

    One PDF per metric is written to
    ``<base_dir>/plots/<datetime>/<plot_name>_<metric>.pdf``, along with a
    ``run_info.txt`` that records the checkpoint name and ready-to-run commands.

    Parameters
    ----------
    all_results:
        Output of ``lsh_retr_sweep.run_sweep`` — keyed by
        ``(lambda_coll_min, linear_every_dim, worker_type)``, value is a list of
        per-subset-size aggregated metric dicts (or ``None`` for failed runs)
        for LSH, and simple aggregated metrics for Random.
    base_dir:
        Root directory of the experiment (``conf.base_dir``).
    dataset_name:
        ``conf.dataset.name`` — used as a prefix in the plot name.
    task_name:
        Full ``conf.task.name`` — the aug-model portion is passed through
        ``convert_name`` to produce the human-readable plot name.
    base_overrides:
        The Hydra override list originally passed to the sweep/plot script.
    corpus_size:
        Total corpus size used to normalise the X-axis to *k/C*.  Inferred
        automatically from ``subset_size=0`` entries when not provided.

    Returns
    -------
    Path
        Directory where the PDF files (and ``run_info.txt``) were saved.
    """
    base_dir = Path(base_dir)
    aug_readable_name = _aug_readable_name(task_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if random_hyper_seed is not None:
        seed_tag = f"rhs{random_hyper_seed}"
    elif trained_loss_type != 'sc_coll_min':
        seed_tag = f"trained_{trained_loss_type}"
    else:
        seed_tag = "trained"
    from reform.utils.script_utils import noise_dim_suffix as _noise_dim_suffix
    nd_sfx = _noise_dim_suffix(noise_sigma_query, noise_sigma_corpus)
    folder_name = f"{aug_readable_name}_{seed_tag}{nd_sfx}_{timestamp}"
    if inner_dir is not None:
        out_dir = base_dir / "plots" / inner_dir / dataset_name / folder_name
    else:
        out_dir = base_dir / "plots" / dataset_name / folder_name

    out_dir.mkdir(parents=True, exist_ok=True)

    plot_name = make_plot_name(dataset_name, task_name, rerank_mode=rerank_mode)
    pkl_name = f"lsh_retr_sweep_{plot_name}.pkl"

    _write_run_info(out_dir, task_name, base_overrides, pkl_name, sweep_script=sweep_script)

    if corpus_size is None:
        corpus_size = _infer_corpus_size(all_results)

    # Separate points by worker type
    lsh_points: list[dict] = [
        entry
        for (_, _, wtype), res in all_results.items()
        if res is not None and wtype == "LSH"
        for entry in res
    ]
    random_points: list[dict] = [
        entry
        for (_, _, wtype), res in all_results.items()
        if res is not None and wtype == "Random"
        for entry in res
    ]

    all_points = lsh_points or random_points
    if not all_points:
        return out_dir

    def _x(points: list[dict]) -> np.ndarray:
        raw = np.array([p["num_evals"] for p in points], dtype=float)
        return raw / corpus_size if (corpus_size is not None and corpus_size > 0) else raw

    available_metrics = [
        (key, label) for key, label in _METRICS if key in all_points[0]
    ]

    for metric_key, metric_label in available_metrics:
        plotly_fig = go.Figure()
        if lsh_points:
            plotly_fig.add_trace(go.Scatter(
                x=_x(lsh_points),
                y=np.array([p[metric_key] for p in lsh_points], dtype=float),
                mode="markers",
                marker={"color": "steelblue", "size": 5, "opacity": 0.5},
                name="LSH",
            ))
        if random_points:
            rx = _x(random_points)
            ry = np.array([p[metric_key] for p in random_points], dtype=float)
            order = np.argsort(rx)
            plotly_fig.add_trace(go.Scatter(
                x=rx[order],
                y=ry[order],
                mode="lines+markers",
                line={"color": "tomato", "width": 2},
                marker={"size": 5},
                name="Random",
            ))
        plotly_fig.update_layout(
            xaxis_title="k/C",
            yaxis_title=metric_label,
            title=f"{metric_label} — {plot_name}",
        )
        safe_key = metric_key.replace("@", "_at_")
        plotly_fig.write_html(out_dir / f"{plot_name}_{safe_key}.html")

    for metric_key, metric_label in available_metrics:
        fig, ax = plt.subplots(figsize=(6, 4))

        if lsh_points:
            ax.scatter(
                _x(lsh_points),
                np.array([p[metric_key] for p in lsh_points], dtype=float),
                alpha=0.4, s=12, linewidths=0, color="steelblue", label="LSH",
            )
        if random_points:
            rx = _x(random_points)
            ry = np.array([p[metric_key] for p in random_points], dtype=float)
            # Sort by x so the line connects points in order
            order = np.argsort(rx)
            ax.plot(rx[order], ry[order], color="tomato", linewidth=1.5,
                    marker="o", markersize=4, label="Random")

        ax.set_xlabel(r"$k/C\ \rightarrow$", fontsize=12)
        ax.set_ylabel(rf"$\mathrm{{{metric_label}}}\ \uparrow$", fontsize=12)
        ax.set_title(f"{metric_label} — {plot_name}", fontsize=9)
        if lsh_points and random_points:
            ax.legend(fontsize=9)

        fig.tight_layout()
        safe_key = metric_key.replace("@", "_at_")
        fig.savefig(out_dir / f"{plot_name}_{safe_key}.pdf", format="pdf")
        plt.close(fig)

    return out_dir


def plot_pos_neg_distributions(
    all_data,
    all_vals,
    cluster_idx,
    out_plot_dir,
    dataset,
    ratio,
    K_val,
    filename_prefix="noisy_rerank",
    xlabel="Augnet cosine similarity",
    name_suffix="",
    include_label_in_filename=True,
):
    """Plot pos/neg cosine-sim histograms for each entry in all_data.

    all_data: list of (label, pos_vals, neg_vals, queries_with_pos, num_queries)
    """
    from reform.utils.script_utils import ratio_suffix as _ratio_suffix

    bins = np.linspace(min(all_vals), max(all_vals), 60)

    for label, pos, neg, queries_with_pos, num_queries in all_data:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(pos, bins=bins, alpha=0.6, label=f"Pos (n={len(pos)})",  density=True, color="C0")
        ax.hist(neg, bins=bins, alpha=0.6, label=f"Neg (n={len(neg)})",  density=True, color="C1")
        ax.axvline(np.mean(pos), color="C0", linestyle="--", linewidth=1.5,
                   label=f"Pos mean={np.mean(pos):.3f}")
        ax.axvline(np.mean(neg), color="C1", linestyle="--", linewidth=1.5,
                   label=f"Neg mean={np.mean(neg):.3f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(f"{xlabel} \u2014 {label} \u2014 Cluster {cluster_idx}")
        ax.legend(loc="upper right")
        ax.text(
            0.98, 0.60,
            f"Oracle pos: {queries_with_pos}/{num_queries} queries",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8),
        )
        plt.tight_layout()
        ratio_sfx    = _ratio_suffix(ratio)
        label_part   = f"_{label}" if include_label_in_filename else ""
        fname = str(Path(out_plot_dir) / f"{filename_prefix}_{dataset}_K={K_val}{ratio_sfx}_cluster{cluster_idx}{label_part}{name_suffix}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved: {fname}")
