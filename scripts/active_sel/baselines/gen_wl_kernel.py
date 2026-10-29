"""
Generate relationship files for the Weisfeiler-Leman (WL) subtree kernel baseline.

For each query, selects the top-K_val corpus items by WL kernel score (cosine
similarity of WL subtree feature vectors), then probes the oracle for positives
and downsamples negatives to maintain the pos:neg ratio.

WL subtree kernel (Shervashidze et al., JMLR 2011):
  phi(G, h) = concatenated label-count histograms for depths 0..h
  score(q, c) = cos_sim(phi(q), phi(c))

Initial node label: degree (since all graphs are structurally unlabeled).
Features are stored as scipy sparse CSR matrices to handle large vocabularies
without materializing the full dense matrix.

The corpus graphs are accessed via train_dataset.corpus_graphs (loaded by GENRunner).

Usage
-----
python scripts/active_sel/baselines/gen_wl_kernel.py --dataset ptc_fr --cluster 3
python scripts/active_sel/baselines/gen_wl_kernel.py --dataset ptc_fr --cluster 3 --K_val 40
python scripts/active_sel/baselines/gen_wl_kernel.py --dataset ptc_fr --cluster 3 --h 4
"""

import argparse
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from across.runner.gen_run import GENRunner
from across.utils.script_utils import (
    K_VAL, CORPUS_SIZE,
    ratio_suffix as _ratio_suffix,
    build_relationships_from_top_k,
    print_relationship_stats,
)

DEFAULT_H = 3  # WL depth


def _build_gnn_runner(dataset: str, cluster: int) -> GENRunner:
    return GENRunner(overrides=[
        f"dataset.name={dataset}",
        "dataset.rel_mode=sub_iso",
        "model.name=GEN",
        "model.scoring_function=cos",
        f"dataset.aug_num_cgraphs={CORPUS_SIZE}",
        "dataset.MinR=0.05",
        "dataset.MaxR=0.25",
        "training.batch_size=8192",
        f"use_cluster={cluster}",
        "task.wandb_enabled=false",
    ])


def _wl_labels_one_iter(graph, labels: dict) -> dict:
    """One round of WL relabeling: new_label[v] = hash(current_label, sorted_nbr_labels)."""
    new_labels = {}
    for v in graph.nodes():
        nbr_labels = sorted(labels[u] for u in graph.neighbors(v))
        new_labels[v] = hash((labels[v], tuple(nbr_labels)))
    return new_labels


def _wl_counters(graphs: list, h: int) -> list[Counter]:
    """Compute per-graph WL label counters across depths 0..h.

    Keys are (depth, wl_label_hash) to avoid cross-depth collisions.
    Initial label is node degree.
    """
    labels_list = [{v: d for v, d in graph.degree()} for graph in graphs]
    counters = [Counter() for _ in graphs]

    for depth in range(h + 1):
        for i, (graph, labels) in enumerate(zip(graphs, labels_list)):
            for v in graph.nodes():
                counters[i][(depth, labels[v])] += 1
        if depth < h:
            labels_list = [
                _wl_labels_one_iter(graph, labels_list[i])
                for i, graph in enumerate(graphs)
            ]

    return counters


def _build_sparse_features(
    counters: list[Counter],
    label_to_idx: dict,
) -> sp.csr_matrix:
    """Convert per-graph counters into a sparse CSR feature matrix [N, vocab]."""
    rows, cols, data = [], [], []
    for i, counter in enumerate(counters):
        for key, cnt in counter.items():
            if key in label_to_idx:
                rows.append(i)
                cols.append(label_to_idx[key])
                data.append(float(cnt))
    n = len(counters)
    vocab = len(label_to_idx)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, vocab), dtype=np.float32)


def _row_normalize(mat: sp.csr_matrix) -> sp.csr_matrix:
    """L2-normalize each row of a sparse matrix."""
    norms = np.sqrt(mat.power(2).sum(axis=1)).A1  # [N]
    norms[norms == 0] = 1.0
    inv_norms = sp.diags(1.0 / norms)
    return inv_norms @ mat


def wl_select(
    query_graphs: list,
    corpus_graphs: list,
    K_val: int,
    h: int,
) -> list[list[int]]:
    """Return top-K_val corpus indices per query by WL cosine similarity."""
    n = len(corpus_graphs)
    K_val = min(K_val, n)

    print(f"Computing WL features (h={h}) for {n} corpus graphs ...")
    c_counters = _wl_counters(corpus_graphs, h)

    # Build global vocabulary from corpus (queries may introduce new labels but
    # those would have zero corpus scores, so we can safely ignore them)
    label_to_idx: dict = {}
    for counter in c_counters:
        for key in counter:
            if key not in label_to_idx:
                label_to_idx[key] = len(label_to_idx)

    print(f"  vocabulary size: {len(label_to_idx)}")
    c_mat = _build_sparse_features(c_counters, label_to_idx)
    c_norm = _row_normalize(c_mat)                  # [n, vocab]

    print(f"Computing WL features for {len(query_graphs)} query graphs ...")
    q_counters = _wl_counters(query_graphs, h)
    q_mat = _build_sparse_features(q_counters, label_to_idx)
    q_norm = _row_normalize(q_mat)                  # [Q, vocab]

    # score[q, c] = dot(q_norm[q], c_norm[c])
    # Result: [Q, n] dense — for Q~500 and n=100k this is 50M floats ≈ 200 MB, fine
    print("Computing WL cosine similarity scores ...")
    scores = (q_norm @ c_norm.T).toarray()          # [Q, n] dense np.float32

    top_k = np.argsort(-scores, axis=1)[:, :K_val].tolist()
    return top_k


def main():
    parser = argparse.ArgumentParser(
        description="Generate WL kernel relationship files for a cluster."
    )
    parser.add_argument("--dataset",  type=str, required=True)
    parser.add_argument("--cluster",  type=int, required=True)
    parser.add_argument("--K_val",    type=int, default=K_VAL,
                        help=f"Budget per query (default: {K_VAL})")
    parser.add_argument("--ratio",    type=float, default=0.13,
                        help="Positive-to-negative ratio for neg downsampling (default: 0.13)")
    parser.add_argument("--h",        type=int, default=DEFAULT_H,
                        help=f"WL iteration depth (default: {DEFAULT_H})")
    args = parser.parse_args()

    # ── dataset / ground truth ────────────────────────────────────────────────
    runner    = _build_gnn_runner(args.dataset, args.cluster)
    runner.set_trainer()
    train_ds  = runner.trainer.train_dataset
    num_queries = len(train_ds.query_graphs)
    num_corpus  = len(train_ds.corpus_graphs)
    q_gts = train_ds.all_gt.reshape(num_queries, num_corpus)
    seed  = runner.conf.training.seed

    assert num_corpus == CORPUS_SIZE, (
        f"Expected {CORPUS_SIZE} corpus graphs, got {num_corpus}"
    )

    # ── WL selection ──────────────────────────────────────────────────────────
    top_k_per_query = wl_select(
        train_ds.query_graphs,
        train_ds.corpus_graphs,
        K_val=args.K_val,
        h=args.h,
    )

    # ── build relationships ───────────────────────────────────────────────────
    print(
        f"\nBuilding relationships: cluster={args.cluster}, K_val={args.K_val}, "
        f"ratio={args.ratio}, h={args.h}"
    )
    relationships, no_pos_count = build_relationships_from_top_k(
        q_gts, top_k_per_query, num_corpus, args.ratio, seed
    )
    print(f"WL: {num_queries - no_pos_count}/{num_queries} queries have >=1 oracle positive")
    print_relationship_stats(relationships, "wl_kernel")

    # ── save ──────────────────────────────────────────────────────────────────
    out_dir = f"data/{args.dataset}/preprocessed/splits/train/clusters/{args.cluster}"
    assert os.path.exists(out_dir), f"Output directory {out_dir} does not exist."

    suffix = f"all_corpus_negs_K={args.K_val}{_ratio_suffix(args.ratio)}_h={args.h}"
    with open(f"{out_dir}/wl_kernel_relationships_{suffix}.pkl", "wb") as f:
        pickle.dump(relationships, f)

    metadata = {"K_val": args.K_val, "ratio": args.ratio, "h": args.h}
    with open(f"{out_dir}/wl_kernel_metadata_{suffix}.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved to {out_dir}/")
    print(f"  wl_kernel_relationships_{suffix}.pkl")
    print(f"  wl_kernel_metadata_{suffix}.pkl")


if __name__ == "__main__":
    main()
