"""
Generate relationship files for the InfoMax baseline.

Adapts InfoMax (Tan et al., ICLR 2025) to query-conditioned corpus selection.
The original paper uses query-agnostic importance scores (EL2N, SSP, etc.).
Here we use I(z) = cos_sim(H_q, H_z) as a query-dependent relevance score,
making selection query-specific.

Objective (per query):
    max_X  sum_z X_z I(z)  -  alpha * sum_{z,s} K_{z,s} X_z X_s
    s.t.   sum_z X_z = K_val,  X_z in {0,1}

Solved via the proximal gradient iterative update (Eq. 13 in the paper):
    X^{t+1} = Softmax(p * I  -  2*p*alpha * K @ X^t)

where p = K_val (the selection budget).  K is the corpus-corpus cosine similarity
matrix, sparsified to the top k_neighbors per row (paper default: k=5).

After T iterations the top-K_val items by X value are selected per query.

Usage
-----
python scripts/active_sel/baselines/gen_infomax.py --dataset ptc_fr --cluster 3
python scripts/active_sel/baselines/gen_infomax.py --dataset ptc_fr --cluster 3 --K_val 40
python scripts/active_sel/baselines/gen_infomax.py --dataset ptc_fr --cluster 3 --alpha 1.0
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

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

# Paper defaults (Section 4.1 / 4.4)
DEFAULT_ALPHA       = 0.3
DEFAULT_T           = 20
DEFAULT_K_NEIGHBORS = 5


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


def _build_sparse_K(c_embeds: torch.Tensor, k_neighbors: int) -> torch.Tensor:
    """Corpus-corpus cosine similarity matrix, top-k_neighbors per row, diagonal zeroed."""
    c_norm = F.normalize(c_embeds, dim=-1)
    K_full = c_norm @ c_norm.T          # [n, n]
    K_full.fill_diagonal_(0.0)

    n = K_full.shape[0]
    k = min(k_neighbors, n - 1)
    K_sparse = torch.zeros_like(K_full)
    topk_vals, topk_idx = torch.topk(K_full, k=k, dim=1)
    K_sparse.scatter_(1, topk_idx, topk_vals)
    return K_sparse                     # [n, n], sparse-like dense, may be asymmetric


def infomax_select(
    q_embeds: torch.Tensor,
    c_embeds: torch.Tensor,
    K_val: int,
    alpha: float,
    T: int,
    k_neighbors: int,
) -> list[list[int]]:
    """Run the InfoMax iterative solver for all queries, return top-K_val indices each."""
    Q = q_embeds.shape[0]
    n = c_embeds.shape[0]
    K_val = min(K_val, n)

    print(f"Building sparse K matrix (k_neighbors={k_neighbors}) ...")
    K = _build_sparse_K(c_embeds, k_neighbors)              # [n, n]

    q_norm = F.normalize(q_embeds, dim=-1)
    c_norm = F.normalize(c_embeds, dim=-1)
    I = q_norm @ c_norm.T                                   # [Q, n]

    X = torch.ones(Q, n) / n                                # [Q, n] uniform init
    p = float(K_val)

    print(f"Running InfoMax solver (T={T}, alpha={alpha}, p={K_val}) ...")
    for t in range(T):
        redundancy = X @ K                                  # [Q, n]
        logits = p * I - 2.0 * p * alpha * redundancy
        X = F.softmax(logits, dim=-1)
        if (t + 1) % 5 == 0:
            print(f"  iteration {t + 1}/{T}")

    top_k = torch.argsort(X, dim=-1, descending=True)[:, :K_val].tolist()
    return top_k


def main():
    parser = argparse.ArgumentParser(
        description="Generate InfoMax relationship files for a cluster."
    )
    parser.add_argument("--dataset",     type=str,   required=True)
    parser.add_argument("--cluster",     type=int,   required=True)
    parser.add_argument("--K_val",       type=int,   default=K_VAL,
                        help=f"Coreset budget per query (default: {K_VAL})")
    parser.add_argument("--ratio",       type=float, default=0.13,
                        help="Positive-to-negative ratio for neg downsampling (default: 0.13)")
    parser.add_argument("--alpha",       type=float, default=DEFAULT_ALPHA,
                        help=f"Pairwise redundancy weight (default: {DEFAULT_ALPHA})")
    parser.add_argument("--T",           type=int,   default=DEFAULT_T,
                        help=f"Solver iterations (default: {DEFAULT_T})")
    parser.add_argument("--k_neighbors", type=int,   default=DEFAULT_K_NEIGHBORS,
                        help=f"Sparse-K neighbours per corpus item (default: {DEFAULT_K_NEIGHBORS})")
    parser.add_argument("--ood-query-dataset", type=str, default=None,
                        help="Legacy OOD: use query embeddings from this dataset; "
                             "corpus and oracle stay with --dataset.")
    parser.add_argument("--src-dataset", type=str, default=None,
                        help="New OOD: run this dataset's GNN on --dataset graphs. "
                             "Corpus/queries/oracle all stay with --dataset.")
    args = parser.parse_args()

    if args.src_dataset and args.ood_query_dataset:
        parser.error("--src-dataset and --ood-query-dataset are mutually exclusive")

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
        emb_path = (
            f"allPklDumps/{args.dataset}/gnn_embs/cluster_{args.cluster}/train/embeddings.pt"
        )
        data = torch.load(emb_path, map_location="cpu")
        c_embeds = data["c_embeds"].float()
        if args.ood_query_dataset:
            ood_emb_path = f"allPklDumps/{args.ood_query_dataset}/gnn_embs/cluster_{args.cluster}/train/embeddings.pt"
            q_embeds = torch.load(ood_emb_path, map_location="cpu")["q_embeds"].float()
            print(f"OOD query embeddings from {args.ood_query_dataset}: {q_embeds.shape[0]} queries")
        else:
            q_embeds = data["q_embeds"].float()

    assert c_embeds.shape[0] == num_corpus, (
        f"Embedding corpus size {c_embeds.shape[0]} != dataset corpus size {num_corpus}"
    )
    print(f"Loaded GNN embeddings: {q_embeds.shape[0]} queries x {c_embeds.shape[0]} corpus")

    # ── InfoMax selection ─────────────────────────────────────────────────────
    top_k_per_query = infomax_select(
        q_embeds, c_embeds,
        K_val=args.K_val,
        alpha=args.alpha,
        T=args.T,
        k_neighbors=args.k_neighbors,
    )

    # ── build relationships ───────────────────────────────────────────────────
    print(
        f"\nBuilding relationships: cluster={args.cluster}, K_val={args.K_val}, "
        f"ratio={args.ratio}, alpha={args.alpha}"
    )
    relationships, no_pos_count = build_relationships_from_top_k(
        q_gts, top_k_per_query, num_corpus, args.ratio, seed
    )
    print(f"InfoMax: {num_queries - no_pos_count}/{num_queries} queries have >=1 oracle positive")
    print_relationship_stats(relationships, "infomax")

    # ── save ──────────────────────────────────────────────────────────────────
    out_dir = f"data/{args.dataset}/preprocessed/splits/train/clusters/{args.cluster}"
    assert os.path.exists(out_dir), f"Output directory {out_dir} does not exist."

    if args.src_dataset:
        ood_sfx = f"_ood_src={args.src_dataset}"
    elif args.ood_query_dataset:
        ood_sfx = f"_ood_query={args.ood_query_dataset}"
    else:
        ood_sfx = ""
    suffix = f"all_corpus_negs_K={args.K_val}{_ratio_suffix(args.ratio)}_alpha={args.alpha}{ood_sfx}"
    with open(f"{out_dir}/infomax_relationships_{suffix}.pkl", "wb") as f:
        pickle.dump(relationships, f)

    metadata = {
        "K_val":             args.K_val,
        "ratio":             args.ratio,
        "alpha":             args.alpha,
        "T":                 args.T,
        "k_neighbors":       args.k_neighbors,
        "ood_query_dataset": args.ood_query_dataset,
        "src_dataset":       args.src_dataset,
    }
    with open(f"{out_dir}/infomax_metadata_{suffix}.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved to {out_dir}/")
    print(f"  infomax_relationships_{suffix}.pkl")
    print(f"  infomax_metadata_{suffix}.pkl")


if __name__ == "__main__":
    main()
