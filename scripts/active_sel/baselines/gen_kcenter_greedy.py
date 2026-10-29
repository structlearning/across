"""
Generate relationship files for the k-center greedy baseline.

Runs greedy k-center selection on the corpus GNN embeddings to find K
maximally-spread representative corpus items. Because the selection is
query-agnostic, the same K centers are used as the candidate set for
every query. The oracle is then probed for positives and negatives are
sampled from the remainder of the corpus (matching gen_dataset_carefully).

Usage
-----
python scripts/active_sel/baselines/gen_kcenter_greedy.py --dataset ptc_fr --cluster 3
python scripts/active_sel/baselines/gen_kcenter_greedy.py --dataset ptc_fr --cluster 3 --K_val 40
python scripts/active_sel/baselines/gen_kcenter_greedy.py --dataset ptc_fr --cluster 3 --ratio 0.13
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import torch

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


def _build_gnn_runner(dataset: str, cluster: int) -> GENRunner:
    # GENRunner is used only to load the train dataset for this cluster
    # and obtain q_gts (ground truth pos/neg labels per query) and the seed.
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


def kcenter_greedy(c_embeds: torch.Tensor, K: int, seed: int) -> list[int]:
    """Greedy k-center on corpus embeddings.

    Starts from a deterministic seed-based initial center, then iteratively
    selects the corpus item farthest from the current set of centers.
    Returns a list of K corpus indices.
    """
    n = c_embeds.shape[0]
    K = min(K, n)

    rng = torch.Generator()
    rng.manual_seed(seed)
    first = torch.randint(n, (1,), generator=rng).item()

    centers = [first]
    # delta[i] = min L2 distance from corpus item i to any selected center
    delta = torch.full((n,), float("inf"))

    first_dists = torch.cdist(c_embeds, c_embeds[first:first+1], p=2).squeeze(1)
    delta = torch.minimum(delta, first_dists)

    for _ in range(K - 1):
        new_center = int(torch.argmax(delta).item())
        centers.append(new_center)
        new_dists = torch.cdist(c_embeds, c_embeds[new_center:new_center+1], p=2).squeeze(1)
        delta = torch.minimum(delta, new_dists)

    return centers


def main():
    parser = argparse.ArgumentParser(
        description="Generate k-center greedy relationship files for a cluster."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g. ptc_fr)")
    parser.add_argument("--cluster", type=int, required=True,
                        help="Cluster index (1-4)")
    parser.add_argument("--K_val", type=int, default=K_VAL,
                        help=f"Number of k-center corpus items (default: {K_VAL})")
    parser.add_argument("--ratio", type=float, default=0.13,
                        help="Positive-to-negative ratio for neg downsampling (default: 0.13)")
    parser.add_argument("--gpu", type=str, default=None,
                        help="CUDA_VISIBLE_DEVICES value (default: inherit from environment)")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # ── dataset / ground truth ────────────────────────────────────────────────
    runner    = _build_gnn_runner(args.dataset, args.cluster)
    runner.set_trainer()
    train_ds  = runner.trainer.train_dataset
    num_queries = len(train_ds.query_graphs)
    num_corpus  = len(train_ds.corpus_graphs)
    q_gts = train_ds.all_gt.reshape(num_queries, num_corpus)
    seed  = runner.conf.training.seed

    # ── GNN embeddings (corpus only needed for k-center) ─────────────────────
    emb_path = f"allPklDumps/{args.dataset}/gnn_embs/cluster_{args.cluster}/train/embeddings.pt"
    data = torch.load(emb_path, map_location="cpu")
    c_embeds = data["c_embeds"].float()
    print(f"Loaded GNN corpus embeddings: {c_embeds.shape[0]} corpus items")

    # ── greedy k-center selection ─────────────────────────────────────────────
    print(f"Running greedy k-center (K={args.K_val}) on {num_corpus} corpus items ...")
    k_center_indices = kcenter_greedy(c_embeds, args.K_val, seed)
    print(f"Selected {len(k_center_indices)} centers")

    # ── build relationships (same centers for every query) ────────────────────
    print(
        f"\nBuilding relationships: cluster={args.cluster}, K_val={args.K_val}, "
        f"ratio={args.ratio}"
    )
    top_k_per_query = [k_center_indices] * num_queries
    relationships, no_pos_count = build_relationships_from_top_k(
        q_gts, top_k_per_query, num_corpus, args.ratio, seed
    )
    print(f"k-center greedy: {num_queries - no_pos_count}/{num_queries} queries have >=1 oracle positive")
    print_relationship_stats(relationships, "k_center_greedy")

    # ── save ──────────────────────────────────────────────────────────────────
    out_dir = f"data/{args.dataset}/preprocessed/splits/train/clusters/{args.cluster}"
    assert os.path.exists(out_dir), f"Output directory {out_dir} does not exist."

    suffix = f"all_corpus_negs_K={args.K_val}{_ratio_suffix(args.ratio)}"
    with open(f"{out_dir}/kcenter_relationships_{suffix}.pkl", "wb") as f:
        pickle.dump(relationships, f)

    metadata = {
        "K_val":          args.K_val,
        "ratio":          args.ratio,
        "k_center_indices": k_center_indices,
    }
    with open(f"{out_dir}/kcenter_metadata_{suffix}.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved to {out_dir}/")
    print(f"  kcenter_relationships_{suffix}.pkl")
    print(f"  kcenter_metadata_{suffix}.pkl")


if __name__ == "__main__":
    main()
