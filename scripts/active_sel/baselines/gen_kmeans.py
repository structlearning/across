"""
Generate relationship files for the k-means baseline.

Clusters the corpus GNN embeddings into K clusters using k-means, then selects
the corpus item closest to each cluster centroid. Because the selection is
query-agnostic, the same K representative items are used as the candidate set
for every query. The oracle is then probed for positives and negatives are
sampled from the remainder of the corpus (matching gen_dataset_carefully).

Usage
-----
python scripts/active_sel/baselines/gen_kmeans.py --dataset ptc_fr --cluster 3
python scripts/active_sel/baselines/gen_kmeans.py --dataset ptc_fr --cluster 3 --K_val 40
python scripts/active_sel/baselines/gen_kmeans.py --dataset ptc_fr --cluster 3 --ratio 0.13
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import torch
from sklearn.cluster import KMeans

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


def kmeans_representatives(c_embeds: torch.Tensor, K: int, seed: int) -> list[int]:
    """Run k-means on corpus embeddings and return the corpus index closest to each centroid.

    Deduplicates in case two centroids snap to the same corpus item.
    """
    K = min(K, c_embeds.shape[0])

    kmeans = KMeans(n_clusters=K, random_state=seed, n_init=10)
    kmeans.fit(c_embeds.numpy())

    centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    dists = torch.cdist(centroids, c_embeds)          # [K, n]
    closest = torch.argmin(dists, dim=1).tolist()

    # preserve order, drop duplicates
    seen = set()
    unique = []
    for idx in closest:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)

    if len(unique) < len(closest):
        print(f"Warning: {len(closest) - len(unique)} centroid(s) snapped to duplicate corpus items; "
              f"using {len(unique)} unique representatives")

    return unique


def main():
    parser = argparse.ArgumentParser(
        description="Generate k-means relationship files for a cluster."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g. ptc_fr)")
    parser.add_argument("--cluster", type=int, required=True,
                        help="Cluster index (1-4)")
    parser.add_argument("--K_val", type=int, default=K_VAL,
                        help=f"Number of k-means clusters / representatives (default: {K_VAL})")
    parser.add_argument("--ratio", type=float, default=0.13,
                        help="Positive-to-negative ratio for neg downsampling (default: 0.13)")
    args = parser.parse_args()

    # ── dataset / ground truth ────────────────────────────────────────────────
    runner    = _build_gnn_runner(args.dataset, args.cluster)
    runner.set_trainer()
    train_ds  = runner.trainer.train_dataset
    num_queries = len(train_ds.query_graphs)
    num_corpus  = len(train_ds.corpus_graphs)
    q_gts = train_ds.all_gt.reshape(num_queries, num_corpus)
    seed  = runner.conf.training.seed

    # ── GNN embeddings (corpus only needed for k-means) ───────────────────────
    emb_path = f"allPklDumps/{args.dataset}/gnn_embs/cluster_{args.cluster}/train/embeddings.pt"
    data = torch.load(emb_path, map_location="cpu")
    c_embeds = data["c_embeds"].float()
    print(f"Loaded GNN corpus embeddings: {c_embeds.shape[0]} corpus items")
    assert c_embeds.shape[0] == num_corpus, (
        f"Embedding corpus size {c_embeds.shape[0]} != dataset corpus size {num_corpus}"
    )

    # ── k-means clustering and centroid snapping ──────────────────────────────
    print(f"Running k-means (K={args.K_val}) on {num_corpus} corpus items ...")
    kmeans_indices = kmeans_representatives(c_embeds, args.K_val, seed)
    print(f"Selected {len(kmeans_indices)} representative corpus items")

    # ── build relationships (same representatives for every query) ────────────
    print(
        f"\nBuilding relationships: cluster={args.cluster}, K_val={args.K_val}, "
        f"ratio={args.ratio}"
    )
    top_k_per_query = [kmeans_indices] * num_queries
    relationships, no_pos_count = build_relationships_from_top_k(
        q_gts, top_k_per_query, num_corpus, args.ratio, seed
    )
    print(f"k-means: {num_queries - no_pos_count}/{num_queries} queries have >=1 oracle positive")
    print_relationship_stats(relationships, "kmeans")

    # ── save ──────────────────────────────────────────────────────────────────
    out_dir = f"data/{args.dataset}/preprocessed/splits/train/clusters/{args.cluster}"
    assert os.path.exists(out_dir), f"Output directory {out_dir} does not exist."

    suffix = f"all_corpus_negs_K={args.K_val}{_ratio_suffix(args.ratio)}"
    with open(f"{out_dir}/kmeans_relationships_{suffix}.pkl", "wb") as f:
        pickle.dump(relationships, f)

    metadata = {
        "K_val":          args.K_val,
        "ratio":          args.ratio,
        "kmeans_indices": kmeans_indices,
    }
    with open(f"{out_dir}/kmeans_metadata_{suffix}.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved to {out_dir}/")
    print(f"  kmeans_relationships_{suffix}.pkl")
    print(f"  kmeans_metadata_{suffix}.pkl")


if __name__ == "__main__":
    main()
