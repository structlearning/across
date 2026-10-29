"""
Plot augnet cosine-sim distributions when a noise dimension is appended to
query and corpus AugNet vectors before LSH construction and reranking.

Unlike noisy_rerank_plot.py (which adds noise to cosine scores), this script
appends one dimension sampled from N(0, σ_q) and N(0, σ_c) to the query and
corpus embeddings respectively (D → D+1), loads the matching D+1 hyperplanes
saved by lsh_retr_sweep.py --noise-sigma-query / --noise-sigma-corpus, retrieves
buckets, reranks by D+1 cosine similarity, and plots the resulting pos/neg
distributions.  σ_q=σ_c=0 is the clean D-dim baseline.

Prerequisites: run lsh_retr_sweep.py with matching --noise-sigma-query and
--noise-sigma-corpus for each (sigma_q, sigma_c) pair you want to plot.

Usage
-----
python scripts/noisy_dim_rerank_plot.py --dataset ptc_fr --cluster 3 \\
    --sigmas-query 0 0.5 1 2 --sigmas-corpus 0 0.5 1 2
python scripts/noisy_dim_rerank_plot.py --dataset ptc_fr --cluster 3 \\
    --sigmas-query 1 2 --sigmas-corpus 0.5 1 --ratio 0.13
"""

import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "misc"))
os.chdir(ROOT)

from wandb_to_human import convert_name
from reform.lsh.lsh import LSH
from reform.utils.script_utils import (
    K_VAL,
    ratio_suffix as _ratio_suffix,
    noise_dim_suffix as _noise_dim_suffix,
    add_aug_args, build_runner, load_aug_checkpoint,
    select_hyperplanes,
    sample_negatives, compute_train_embeddings, cosine_sims_for_relationships,
)
from reform.utils.lsh_plots import plot_pos_neg_distributions


# ── helpers ───────────────────────────────────────────────────────────────────

def _append_noise(embs, sigma, rng):
    """Append one noise dimension N(0, sigma) to embeddings [N, G, D] → [N, G, D+1]."""
    noise = torch.tensor(
        rng.normal(0, sigma, (*embs.shape[:-1], 1)),
        dtype=embs.dtype,
        device=embs.device,
    )
    return torch.cat([embs, noise], dim=-1)


def _noisy_cosine_sims(q_embs, c_embs):
    """D+1 cosine sims from the first gen_embd_dim slice."""
    q_normed = F.normalize(q_embs[:, 0, :], dim=-1)
    c_normed = F.normalize(c_embs[:, 0, :], dim=-1)
    return q_normed @ c_normed.T   # [Q, C]


# ── main pipeline ─────────────────────────────────────────────────────────────

def run_noisy_dim_mode(
    cluster_runner, conf, ratio, cluster_idx, sigma_pairs, out_plot_dir,
    out_base, dataset, augnet_name,
    K_val=40, k_over_c_min=0.2, k_over_c_max=0.6,
    noise_seed=0, name_suffix="",
):
    out_dir = f"{out_base}/{cluster_idx}/{augnet_name}"
    out_plot_dir = os.path.join(out_plot_dir, augnet_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_plot_dir, exist_ok=True)

    query_aug_embs, corpus_aug_embs, q_gts = compute_train_embeddings(cluster_runner)
    num_queries = query_aug_embs.shape[0]
    num_corpus  = corpus_aug_embs.shape[0]

    all_data = []
    all_vals = []

    for sigma_q, sigma_c in sigma_pairs:
        noise_rng = np.random.default_rng(noise_seed)

        if sigma_q > 0:
            q_embs = _append_noise(query_aug_embs, sigma_q, noise_rng)
        else:
            q_embs = query_aug_embs

        if sigma_c > 0:
            c_embs = _append_noise(corpus_aug_embs, sigma_c, noise_rng)
        else:
            c_embs = corpus_aug_embs

        cosine_sims = _noisy_cosine_sims(q_embs, c_embs)

        hp  = select_hyperplanes(
            dataset, augnet_name,
            k_over_c_min=k_over_c_min, k_over_c_max=k_over_c_max,
            noise_sigma_query=sigma_q, noise_sigma_corpus=sigma_c,
        )
        lsh = LSH(
            torch.from_numpy(hp["hp_weights"]),
            torch.from_numpy(hp["hp_biases"]),
            hp["bit_indices"],
            c_embs.to("cpu").permute(1, 0, 2),   # [G, C, D(+1)]
        )
        buckets = [sorted(lsh.retrieve(q_embs[i].cpu())) for i in range(num_queries)]

        relationships = {}
        no_pos_count  = 0

        for local_idx, (gt, c_ids) in enumerate(zip(q_gts, buckets)):
            scores      = cosine_sims[local_idx][c_ids]
            top_k_idxs  = torch.argsort(scores, descending=True)[:K_val]
            top_k_c_ids = [c_ids[i] for i in top_k_idxs.tolist()]

            pos_c_ids = [cid for cid in top_k_c_ids if gt[cid]]
            if not pos_c_ids:
                no_pos_count += 1
                pos_c_ids = [0]

            neg_c_ids = sorted(set(range(num_corpus)) - set(pos_c_ids))
            neg_c_ids = sample_negatives(
                neg_c_ids, ratio, pos_c_ids, conf.training.seed + local_idx
            )
            relationships[local_idx] = {"pos": pos_c_ids, "neg": neg_c_ids}

        nd_sfx   = _noise_dim_suffix(sigma_q, sigma_c)
        suffix   = f"all_corpus_negs_K={K_val}{_ratio_suffix(ratio)}{nd_sfx}{name_suffix}"
        pkl_path = f"{out_dir}/lsh_relationships_{suffix}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(relationships, f)

        metadata = {
            "K_val":              K_val,
            "aug_ckpt_tag":       cluster_runner.conf.task.name,
            "ratio":              ratio,
            "noise_sigma_query":  sigma_q,
            "noise_sigma_corpus": sigma_c,
            "hp_weights":         hp["hp_weights"],
            "hp_biases":          hp["hp_biases"],
            "bit_indices":        hp["bit_indices"],
            "noise_seed":         noise_seed,
        }
        with open(f"{out_dir}/lsh_metadata_{suffix}.pkl", "wb") as f:
            pickle.dump(metadata, f)

        pos_vals, neg_vals, queries_with_pos = cosine_sims_for_relationships(
            relationships, cosine_sims, q_gts
        )
        print(
            f"sq={sigma_q} sc={sigma_c}: pos n={len(pos_vals)}  mean={np.mean(pos_vals):.4f} | "
            f"neg n={len(neg_vals)}  mean={np.mean(neg_vals):.4f} | "
            f"no_pos={no_pos_count}/{num_queries}"
        )

        label = f"noisy dim \u2014 \u03c3_q={sigma_q} \u03c3_c={sigma_c}"
        all_data.append((label, pos_vals, neg_vals, queries_with_pos, num_queries))
        all_vals.extend(pos_vals + neg_vals)

    plot_pos_neg_distributions(
        all_data, all_vals, cluster_idx, out_plot_dir,
        dataset=dataset, ratio=ratio, K_val=K_val,
        filename_prefix="noisy_dim",
        xlabel="Augnet cosine similarity (D+1)",
        name_suffix=name_suffix,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def _sigma_val(x):
    v = float(x)
    return int(v) if v == int(v) else v


def main():
    parser = argparse.ArgumentParser(
        description="Plot augnet cosine-sim distributions under noisy-dimension LSH reranking."
    )
    parser.add_argument("--dataset",  type=str, required=True)
    parser.add_argument("--cluster",  type=int, required=True, help="Cluster index (1-4)")
    parser.add_argument("--gpu",      type=str, default=None)
    parser.add_argument(
        "--sigmas-query", type=_sigma_val, nargs="+", required=True,
        help="Query noise sigma values (one per (sigma_q, sigma_c) pair)",
    )
    parser.add_argument(
        "--sigmas-corpus", type=_sigma_val, nargs="+", required=True,
        help="Corpus noise sigma values (must be the same length as --sigmas-query)",
    )
    parser.add_argument("--ratio",  type=float, default=0.13,
                        help="Pos:neg ratio for negative downsampling (default: 0.13)")
    parser.add_argument("--K_val",  type=int, default=K_VAL,
                        help=f"Top-K rerank window (default: {K_VAL})")
    parser.add_argument("--noise-seed", type=int, default=0,
                        help="Seed for noise dimension sampling (default: 0)")
    parser.add_argument("--out_plots",  type=str, default="plots/noisy_dim_rerank",
                        help="Directory to save plot PNGs (default: plots/noisy_dim_rerank)")
    add_aug_args(parser)
    args = parser.parse_args()

    if len(args.sigmas_query) != len(args.sigmas_corpus):
        parser.error("--sigmas-query and --sigmas-corpus must have the same number of values")

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    out_base = f"data/{args.dataset}/preprocessed/splits/train/clusters"

    cluster_runner = build_runner(args)
    augnet_name    = convert_name(cluster_runner.conf.task.name)

    load_aug_checkpoint(cluster_runner)

    sigma_pairs = list(zip(args.sigmas_query, args.sigmas_corpus))
    print(
        f"Running noisy-dim LSH for cluster {args.cluster}, "
        f"sigma pairs (q, c)={sigma_pairs}, ratio={args.ratio}"
    )
    run_noisy_dim_mode(
        cluster_runner, cluster_runner.conf,
        ratio=args.ratio,
        cluster_idx=args.cluster,
        sigma_pairs=sigma_pairs,
        out_plot_dir=args.out_plots,
        out_base=out_base,
        dataset=args.dataset,
        augnet_name=augnet_name,
        K_val=args.K_val,
        k_over_c_min=args.k_over_c_min,
        k_over_c_max=args.k_over_c_max,
        noise_seed=args.noise_seed,
    )


if __name__ == "__main__":
    main()
