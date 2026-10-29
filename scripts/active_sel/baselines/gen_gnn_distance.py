"""
Generate relationship files for the GNN-distance baseline.

For each query, selects the top-K_val corpus items by smallest distance
on pre-computed GNN embeddings, then probes the oracle for positives and samples
negatives from the remainder of the corpus (matching gen_dataset_carefully).

Distance metrics
----------------
l2     (default): Euclidean distance via torch.cdist
l1               : Manhattan distance via torch.cdist
hinge            : ||relu(H_q - H_c)||_1 — asymmetric, captures how much the
                   corpus item falls short of the query in each embedding dim.
                   Computed in query-chunks to avoid materialising [Q x n x d].
                   NOTE: hinge has no FAISS counterpart; --faiss is not supported.

Retrieval modes
---------------
Exact (default): full pairwise torch.cdist argsort — precise but O(Q*n*d).
FAISS IVF (--faiss): approximate IVF index, much faster for large corpora.
  nlist is set to sqrt(n); nprobe defaults to 16.
  Only supported for l1 and l2; incompatible with hinge.

Usage
-----
python scripts/active_sel/baselines/gen_gnn_distance.py --dataset ptc_fr --cluster 3
python scripts/active_sel/baselines/gen_gnn_distance.py --dataset ptc_fr --cluster 3 --dist l1
python scripts/active_sel/baselines/gen_gnn_distance.py --dataset ptc_fr --cluster 3 --dist hinge
python scripts/active_sel/baselines/gen_gnn_distance.py --dataset ptc_fr --cluster 3 --ratio 0.13
python scripts/active_sel/baselines/gen_gnn_distance.py --dataset ptc_fr --cluster 3 --faiss
python scripts/active_sel/baselines/gen_gnn_distance.py --dataset ptc_fr --cluster 3 --faiss --nprobe 32
"""

import argparse
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from across.runner.gen_run import GENRunner
from across.utils.model_utils import compute_gnn_embeds
from across.utils.utils import get_class
from across.utils.script_utils import (
    K_VAL, CORPUS_SIZE,
    ratio_suffix as _ratio_suffix,
    build_relationships_from_top_k,
    print_relationship_stats,
)


def _compute_src_gnn_embeds(runner: GENRunner, src_dataset: str):
    """Run src GNN on target dataset graphs; returns (q_embeds, c_embeds) on CPU."""
    conf = runner.conf
    src_ckpt_tag = (
        f"GEN_{src_dataset}_numC={CORPUS_SIZE}_MinR=0.05_MaxR=0.25"
        "_scoring_function=cos,stemp=0.1,margin=1.0"
    )
    ckpt_path = f"bestValidationModels/{src_ckpt_tag}/best.pt"
    model = get_class(f"{conf.model.classPath}.{conf.model.name}")(
        conf, runner.gmn_config
    ).to(conf.training.device)
    ckpt = torch.load(ckpt_path, map_location=conf.training.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded src GNN from {ckpt_path}")
    q_embeds, c_embeds = compute_gnn_embeds(model, runner.trainer.train_dataset, pre_last_layer=False)
    return q_embeds.float().cpu(), c_embeds.float().cpu()


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


def _top_k_exact(q_embeds: torch.Tensor, c_embeds: torch.Tensor, p: int, K: int,
                 noise_sigma: float = 0.0, seed: int = 0) -> list[list[int]]:
    dists = torch.cdist(q_embeds, c_embeds, p=p)
    if noise_sigma > 0.0:
        gen = torch.Generator()
        gen.manual_seed(seed)
        dists = dists + torch.randn(dists.shape, generator=gen) * noise_sigma
    return torch.argsort(dists, dim=1)[:, :K].tolist()


def _top_k_hinge(q_embeds: torch.Tensor, c_embeds: torch.Tensor, K: int,
                 chunk_size: int = 64, noise_sigma: float = 0.0, seed: int = 0) -> list[list[int]]:
    # Materialising [Q, n, d] at once is ~128 GB for Q=5k, n=100k, d=64.
    # Process queries in chunks to keep peak memory to chunk_size * n * d floats.
    gen = torch.Generator()
    gen.manual_seed(seed)
    results = []
    for i in range(0, q_embeds.shape[0], chunk_size):
        q_chunk = q_embeds[i:i + chunk_size]                          # [chunk, d]
        diffs = q_chunk.unsqueeze(1) - c_embeds.unsqueeze(0)          # [chunk, n, d]
        hinge_dists = torch.relu(diffs).sum(dim=-1)                   # [chunk, n]
        if noise_sigma > 0.0:
            hinge_dists = hinge_dists + torch.randn(hinge_dists.shape, generator=gen) * noise_sigma
        results.append(torch.argsort(hinge_dists, dim=1)[:, :K])
    return torch.cat(results, dim=0).tolist()


def _top_k_faiss(q_embeds: torch.Tensor, c_embeds: torch.Tensor, dist: str,
                 K: int, nprobe: int) -> list[list[int]]:
    import faiss

    n, d = c_embeds.shape
    nlist = int(math.sqrt(n))

    metric = faiss.METRIC_L1 if dist == "l1" else faiss.METRIC_L2
    quantizer = faiss.IndexFlat(d, metric)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, metric)
    index.nprobe = nprobe

    c_np = c_embeds.numpy().astype(np.float32)
    q_np = q_embeds.numpy().astype(np.float32)

    index.train(c_np)
    index.add(c_np)

    _, indices = index.search(q_np, K)
    return indices.tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Generate GNN-distance relationship files for a cluster."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g. ptc_fr)")
    parser.add_argument("--cluster", type=int, required=True,
                        help="Cluster index (1-4)")
    parser.add_argument("--K_val", type=int, default=K_VAL,
                        help=f"Number of nearest corpus items per query (default: {K_VAL})")
    parser.add_argument("--ratio", type=float, default=0.13,
                        help="Positive-to-negative ratio for neg downsampling (default: 0.13)")
    parser.add_argument("--dist", type=str, default="l2", choices=["l1", "l2", "hinge"],
                        help="Distance metric: l1, l2, or hinge=||relu(H_q-H_c)||_1 (default: l2)")
    parser.add_argument("--faiss", action="store_true", default=False,
                        help="Use approximate FAISS IVF index instead of exact torch.cdist "
                             "(only supported for l1 and l2, not hinge)")
    parser.add_argument("--nprobe", type=int, default=16,
                        help="[--faiss only] Number of IVF cells to search at query time (default: 16)")
    parser.add_argument("--ood-query-dataset", type=str, default=None,
                        help="Legacy OOD: use query embeddings from this dataset; "
                             "corpus and oracle stay with --dataset.")
    parser.add_argument("--src-dataset", type=str, default=None,
                        help="New OOD: run this dataset's GNN on --dataset graphs. "
                             "Corpus/queries/oracle all stay with --dataset.")
    parser.add_argument("--noise-sigma", type=float, default=0.0,
                        help="Std-dev of Gaussian noise added to distances before top-K selection "
                             "(default: 0.0 = no noise). Incompatible with --faiss.")
    args = parser.parse_args()

    if args.faiss and args.dist == "hinge":
        parser.error("--faiss is not supported for --dist hinge (no FAISS hinge metric)")
    if args.src_dataset and args.ood_query_dataset:
        parser.error("--src-dataset and --ood-query-dataset are mutually exclusive")
    if args.faiss and args.noise_sigma > 0.0:
        parser.error("--noise-sigma is not supported with --faiss")

    # ── dataset / ground truth ────────────────────────────────────────────────
    runner    = _build_gnn_runner(args.dataset, args.cluster)
    runner.set_trainer()
    train_ds  = runner.trainer.train_dataset
    num_queries = len(train_ds.query_graphs)
    num_corpus  = len(train_ds.corpus_graphs)
    q_gts = train_ds.all_gt.reshape(num_queries, num_corpus)
    seed  = runner.conf.training.seed

    # ── GNN embeddings ────────────────────────────────────────────────────────
    if args.src_dataset:
        print(f"New OOD: running {args.src_dataset} GNN on {args.dataset} graphs ...")
        q_embeds, c_embeds = _compute_src_gnn_embeds(runner, args.src_dataset)
    else:
        emb_path = f"allPklDumps/{args.dataset}/gnn_embs/cluster_{args.cluster}/train/embeddings.pt"
        data = torch.load(emb_path, map_location="cpu")
        c_embeds = data["c_embeds"].float()
        if args.ood_query_dataset:
            ood_emb_path = f"allPklDumps/{args.ood_query_dataset}/gnn_embs/cluster_{args.cluster}/train/embeddings.pt"
            q_embeds = torch.load(ood_emb_path, map_location="cpu")["q_embeds"].float()
            print(f"OOD query embeddings from {args.ood_query_dataset}: {q_embeds.shape[0]} queries")
        else:
            q_embeds = data["q_embeds"].float()

    print(f"Loaded GNN embeddings: {q_embeds.shape[0]} queries x {c_embeds.shape[0]} corpus")

    # ── top-K retrieval ───────────────────────────────────────────────────────
    noise_tag = f", noise_sigma={args.noise_sigma}" if args.noise_sigma > 0.0 else ""
    if args.dist == "hinge":
        print(f"Computing hinge distances ||relu(H_q - H_c)||_1 (chunked){noise_tag} ...")
        top_k_indices = _top_k_hinge(q_embeds, c_embeds, args.K_val,
                                     noise_sigma=args.noise_sigma, seed=seed)
    elif args.faiss:
        nlist = int(math.sqrt(c_embeds.shape[0]))
        print(f"Building FAISS IVF index (nlist={nlist}, nprobe={args.nprobe}, dist={args.dist}) ...")
        top_k_indices = _top_k_faiss(q_embeds, c_embeds, args.dist, args.K_val, args.nprobe)
    else:
        p = 1 if args.dist == "l1" else 2
        print(f"Computing exact pairwise L{p} distances{noise_tag} ...")
        top_k_indices = _top_k_exact(q_embeds, c_embeds, p, args.K_val,
                                     noise_sigma=args.noise_sigma, seed=seed)

    # ── build relationships ───────────────────────────────────────────────────
    print(
        f"\nBuilding relationships: cluster={args.cluster}, K_val={args.K_val}, "
        f"ratio={args.ratio}, dist={args.dist}, faiss={args.faiss}"
    )
    relationships, no_pos_count = build_relationships_from_top_k(
        q_gts, top_k_indices, num_corpus, args.ratio, seed
    )
    print(f"GNN-distance: {num_queries - no_pos_count}/{num_queries} queries have >=1 oracle positive")
    print_relationship_stats(relationships, f"gnn_distance ({args.dist})")

    # ── save ──────────────────────────────────────────────────────────────────
    out_dir = f"data/{args.dataset}/preprocessed/splits/train/clusters/{args.cluster}"
    assert os.path.exists(out_dir), f"Output directory {out_dir} does not exist."

    faiss_suffix = f"_faiss_nprobe={args.nprobe}" if args.faiss else ""
    noise_suffix = f"_noise_sigma={args.noise_sigma}" if args.noise_sigma > 0.0 else ""
    if args.src_dataset:
        ood_sfx = f"_ood_src={args.src_dataset}"
    elif args.ood_query_dataset:
        ood_sfx = f"_ood_query={args.ood_query_dataset}"
    else:
        ood_sfx = ""
    suffix = f"all_corpus_negs_K={args.K_val}{_ratio_suffix(args.ratio)}_dist={args.dist}{noise_suffix}{faiss_suffix}{ood_sfx}"
    with open(f"{out_dir}/gnn_dist_relationships_{suffix}.pkl", "wb") as f:
        pickle.dump(relationships, f)

    metadata = {
        "K_val":             args.K_val,
        "ratio":             args.ratio,
        "dist":              args.dist,
        "noise_sigma":       args.noise_sigma,
        "faiss":             args.faiss,
        "nprobe":            args.nprobe if args.faiss else None,
        "ood_query_dataset": args.ood_query_dataset,
        "src_dataset":       args.src_dataset,
    }
    with open(f"{out_dir}/gnn_distance_metadata_{suffix}.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved to {out_dir}/")
    print(f"  gnn_dist_relationships_{suffix}.pkl")
    print(f"  gnn_distance_metadata_{suffix}.pkl")


if __name__ == "__main__":
    main()
