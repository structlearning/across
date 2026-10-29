"""
Plot augnet cosine-sim distributions under noisy top-K reranking (LSH) or
random corpus sampling for a single cluster.

LSH mode generates and saves relationship files (reproducible — same inputs
always produce the same files). Random mode loads pre-generated relationship
files produced by gen_dataset_carefully.py to avoid accidental overwrites.

Usage
-----
# LSH mode (default) — generates lsh_relationships_*_noise_sigma=X.pkl
python scripts/noisy_rerank_plot.py --dataset ptc_fr --cluster 3
python scripts/noisy_rerank_plot.py --dataset ptc_fr --cluster 3 --gpu 0 --sigmas 0 0.1 0.5 1 2 5 10 100
python scripts/noisy_rerank_plot.py --dataset ptc_fr --cluster 3 --ratio 0.13

# Random mode — loads random_relationships_*.pkl (generate first with gen_dataset_carefully.py)
python scripts/noisy_rerank_plot.py --dataset ptc_fr --cluster 3 --mode random
python scripts/noisy_rerank_plot.py --dataset ptc_fr --cluster 3 --mode random --ratio 0.13
"""

import argparse
import copy
import os
import sys
import pickle

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn.functional as F

# ── project root on path ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "misc"))
os.chdir(ROOT)

from wandb_to_human import convert_name
from reform.lsh.lsh import LSH
from reform.utils.utils import get_class
from reform.utils.script_utils import (
    K_VAL,
    ratio_suffix as _ratio_suffix,
    add_aug_args, build_runner, build_random_init_runner, build_ood_query_runner, load_aug_checkpoint,
    add_fmap_args, build_fmap_runner, compute_fmap_train_embeddings,
    select_hyperplanes, select_trained_hyperplanes,
    sample_negatives, compute_train_embeddings, cosine_sims_for_relationships,
)
from reform.utils.lsh_plots import plot_pos_neg_distributions


# ── shared embedding computation ──────────────────────────────────────────────
def _compute_embeddings(cluster_runner, query_graph_model=None, corpus_graph_model=None, override_W0_q=None):
    query_aug_embs, corpus_aug_embs, q_gts = compute_train_embeddings(
        cluster_runner, query_graph_model=query_graph_model,
        corpus_graph_model=corpus_graph_model, override_W0_q=override_W0_q,
    )
    cosine_sims = (
        F.normalize(query_aug_embs[:, 0, :], dim=-1) @
        F.normalize(corpus_aug_embs[:, 0, :], dim=-1).T
    ).cpu()
    return cosine_sims, q_gts, query_aug_embs, corpus_aug_embs


# ── LSH mode ──────────────────────────────────────────────────────────────────
def run_lsh_mode(hp, cluster_runner, conf, ratio, cluster_idx, sigmas, out_plot_dir,
                 out_base, dataset, augnet_name,
                 K_val=40, overwrite=False, name_suffix="", query_graph_model=None,
                 corpus_graph_model=None, override_W0_q=None, precomputed_embs=None,
                 replica_split=False):
    out_dir = f"{out_base}/{cluster_idx}/{augnet_name}"
    out_plot_dir = os.path.join(out_plot_dir, augnet_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_plot_dir, exist_ok=True)

    all_data = []
    all_vals = []

    if precomputed_embs is not None:
        _q, _c, _gt = precomputed_embs
        _cs = (F.normalize(_q[:, 0, :], dim=-1) @ F.normalize(_c[:, 0, :], dim=-1).T).cpu()
        _default_embs = (_cs, _gt, _q, _c)
    else:
        _default_embs = None

    if not overwrite:
        if _default_embs is not None:
            cosine_sims, q_gts, _, _ = _default_embs
        else:
            cosine_sims, q_gts, _, _ = _compute_embeddings(cluster_runner, query_graph_model=query_graph_model, corpus_graph_model=corpus_graph_model, override_W0_q=override_W0_q)
        num_queries        = cosine_sims.shape[0]
        for sigma in sigmas:
            suffix   = f"all_corpus_negs_K={K_val}{_ratio_suffix(ratio)}_noise_sigma={sigma}{name_suffix}"
            pkl_path = f"{out_dir}/lsh_relationships_{suffix}.pkl"
            print(f"Loading {pkl_path}")
            with open(pkl_path, "rb") as f:
                relationships = pickle.load(f)
            pos_vals, neg_vals, queries_with_pos = cosine_sims_for_relationships(
                relationships, cosine_sims, q_gts
            )
            print(
                f"sigma={sigma}: pos n={len(pos_vals)}  mean={np.mean(pos_vals):.4f} | "
                f"neg n={len(neg_vals)}  mean={np.mean(neg_vals):.4f}"
            )
            label = f"noisy top-{K_val} rerank \u2014 \u03c3={sigma}"
            all_data.append((label, pos_vals, neg_vals, queries_with_pos, num_queries))
            all_vals.extend(pos_vals + neg_vals)
    else:
        # LSH index always built from trained corpus embeddings
        if _default_embs is not None:
            cosine_sims, q_gts, query_aug_embs, corpus_aug_embs = _default_embs
        else:
            cosine_sims, q_gts, query_aug_embs, corpus_aug_embs = _compute_embeddings(
                cluster_runner, query_graph_model=query_graph_model, override_W0_q=override_W0_q,
            )
        # cosine scores use the (possibly random-init) corpus when --random-init-gm-c is set
        if corpus_graph_model is not None:
            cosine_sims, _, _, _ = _compute_embeddings(
                cluster_runner, query_graph_model=query_graph_model,
                corpus_graph_model=corpus_graph_model, override_W0_q=override_W0_q,
            )
        num_queries = cosine_sims.shape[0]
        num_corpus  = cosine_sims.shape[1]

        if replica_split:
            R = conf.aug_embedder.num_hyperplanes
            N_c, D, K = corpus_aug_embs.shape
            embd_dim = K // R
            c_lsh = corpus_aug_embs.to("cpu").view(N_c, D, R, embd_dim).view(N_c, D * R, embd_dim).permute(1, 0, 2)
            lsh = LSH(torch.from_numpy(hp["hp_weights"]), torch.from_numpy(hp["hp_biases"]), hp["bit_indices"], c_lsh)
            buckets = [sorted(lsh.retrieve(q.cpu().view(D, R, embd_dim).view(D * R, embd_dim))) for q in query_aug_embs]
        else:
            lsh = LSH(
                torch.from_numpy(hp["hp_weights"]),
                torch.from_numpy(hp["hp_biases"]),
                hp["bit_indices"],
                corpus_aug_embs.to("cpu").permute(1, 0, 2),
            )
            buckets = [sorted(lsh.retrieve(query.cpu())) for query in query_aug_embs]

        for sigma in sigmas:
            noise_rng        = np.random.default_rng(0)
            lsh_pos, lsh_neg = [], []
            relationships    = {}
            no_pos_count     = 0

            for local_idx, (gt, c_ids) in enumerate(zip(q_gts, buckets)):
                scores = cosine_sims[local_idx][c_ids]

                if sigma > 0:
                    noise        = torch.tensor(
                        noise_rng.normal(0, sigma, len(c_ids)), dtype=scores.dtype
                    )
                    noisy_scores = scores + noise
                else:
                    noisy_scores = scores

                top_k_idxs  = torch.argsort(noisy_scores, descending=True)[:K_val]
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
                for cid in pos_c_ids:
                    lsh_pos.append(cosine_sims[local_idx][cid].item())
                for cid in neg_c_ids:
                    lsh_neg.append(cosine_sims[local_idx][cid].item())

            print(
                f"sigma={sigma}: pos n={len(lsh_pos)}  mean={np.mean(lsh_pos):.4f} | "
                f"neg n={len(lsh_neg)}  mean={np.mean(lsh_neg):.4f}"
            )

            suffix = f"all_corpus_negs_K={K_val}{_ratio_suffix(ratio)}_noise_sigma={sigma}{name_suffix}"
            with open(f"{out_dir}/lsh_relationships_{suffix}.pkl", "wb") as f:
                pickle.dump(relationships, f)

            metadata = {
                "hp_weights":   hp["hp_weights"],
                "hp_biases":    hp["hp_biases"],
                "bit_indices":  hp["bit_indices"],
                "K_val":        K_val,
                "aug_ckpt_tag": cluster_runner.conf.task.name,
                "ratio":        ratio,
                "noise_sigma":  sigma,
            }
            with open(f"{out_dir}/lsh_metadata_{suffix}.pkl", "wb") as f:
                pickle.dump(metadata, f)
            print(f"  Saved to {out_dir}/")

            queries_with_pos = num_queries - no_pos_count
            label = f"noisy top-{K_val} rerank \u2014 \u03c3={sigma}"
            all_data.append((label, lsh_pos, lsh_neg, queries_with_pos, num_queries))
            all_vals.extend(lsh_pos + lsh_neg)

    plot_pos_neg_distributions(all_data, all_vals, cluster_idx, out_plot_dir,
                               dataset=dataset, ratio=ratio, K_val=K_val,
                               filename_prefix="noisy_rerank", name_suffix=name_suffix)


# ── Random mode ───────────────────────────────────────────────────────────────
def plot_random_mode(cluster_runner, ratio, cluster_idx, out_plot_dir,
                     out_base, dataset,
                     K_val=40, name_suffix="",
                     query_graph_model=None, corpus_graph_model=None, override_W0_q=None,
                     precomputed_embs=None):
    if precomputed_embs is not None:
        _q, _c, q_gts = precomputed_embs
        cosine_sims = (F.normalize(_q[:, 0, :], dim=-1) @ F.normalize(_c[:, 0, :], dim=-1).T).cpu()
    else:
        cosine_sims, q_gts, _, _ = _compute_embeddings(cluster_runner, query_graph_model=query_graph_model, corpus_graph_model=corpus_graph_model, override_W0_q=override_W0_q)
    num_queries        = cosine_sims.shape[0]

    out_dir  = f"{out_base}/{cluster_idx}"
    suffix   = f"all_corpus_negs_K={K_val}{_ratio_suffix(ratio)}"
    pkl_path = f"{out_dir}/random_relationships_{suffix}.pkl"
    print(f"Loading {pkl_path}")
    with open(pkl_path, "rb") as f:
        relationships = pickle.load(f)

    os.makedirs(out_plot_dir, exist_ok=True)

    pos_vals, neg_vals, queries_with_pos = cosine_sims_for_relationships(
        relationships, cosine_sims, q_gts
    )
    print(
        f"random K={K_val}: pos n={len(pos_vals)}  mean={np.mean(pos_vals):.4f} | "
        f"neg n={len(neg_vals)}  mean={np.mean(neg_vals):.4f}"
    )

    label = f"random K={K_val}"
    plot_pos_neg_distributions(
        [(label, pos_vals, neg_vals, queries_with_pos, num_queries)],
        pos_vals + neg_vals, cluster_idx, out_plot_dir,
        dataset=dataset, ratio=ratio, K_val=K_val,
        filename_prefix="random", include_label_in_filename=False, name_suffix=name_suffix,
    )


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Plot augnet cosine-sim distributions from pre-generated relationship pickles."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g. ptc_fr)")
    parser.add_argument("--cluster", type=int, required=True,
                        help="Cluster index (1-4)")
    parser.add_argument("--gpu", type=str, default=None,
                        help="CUDA_VISIBLE_DEVICES value (default: inherit from environment)")
    parser.add_argument("--mode", type=str, default="lsh", choices=["lsh", "random"],
                        help="Relationship mode: 'lsh' (noisy rerank) or 'random' (default: lsh)")
    parser.add_argument("--sigmas", type=lambda x: int(x) if float(x) == int(float(x)) else float(x), nargs="+",
                        default=[0, 0.5, 0.1, 1, 2, 5, 10, 100],
                        help="[LSH only] Sigma values to load and plot (default: 0 0.1 0.5 1 2 5 10 100)")
    parser.add_argument("--ratio", type=float, default=None,
                        help="Ratio used when generating the relationship files (determines filename). "
                             "Omit if files were generated without a ratio.")
    parser.add_argument("--out_plots", type=str,
                        default="plots/noisy_rerank",
                        help="Directory to save plot PNGs (default: plots/noisy_rerank)")
    parser.add_argument("--K_val", type=int, default=K_VAL,
                        help=f"Number of corpus items to retrieve/sample per query (default: {K_VAL})")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="[LSH only] Regenerate and overwrite relationship/metadata files (default: load existing)")
    parser.add_argument("--trained-hyper", action="store_true", default=False,
                        help="[LSH only] Use trained LSH hyperplanes instead of random hyperplanes")
    parser.add_argument("--third-loss-type", type=str, default="sc_coll_min",
                        choices=["sc_coll_min", "ad_decorr"],
                        help="[LSH only, --trained-hyper] Third loss type used during LSH training (default: sc_coll_min)")
    parser.add_argument("--subtract-mean", action="store_true", default=False,
                        help="[LSH only, random hyperplanes] Select from sweep file generated with --subtract-mean (default: False)")
    parser.add_argument("--random-init-gm-q", action="store_true", default=False,
                        help="Use a randomly initialized query GNN instead of the trained graph_model_best (affects both long and short)")
    parser.add_argument("--random-init-w0-q", action="store_true", default=False,
                        help="Override W_0 in q_hat (long form only) with last-layer params from a freshly initialized GNN")
    parser.add_argument("--random-init-gm-c", action="store_true", default=False,
                        help="[AugNet only] Use a randomly initialized corpus GNN for cosine scoring")
    parser.add_argument("--use-fmap", action="store_true", default=False,
                        help="Use FMap embeddings instead of AugNet embeddings")
    parser.add_argument("--pre-augnet", action="store_true", default=False,
                        help="Use pre-AugNet (random-feature-map) embeddings instead of AugNet MLP outputs")
    parser.add_argument("--replica-split", action="store_true", default=False,
                        help="Implies --pre-augnet; each hyperplane gets its own hash table. "
                             "Uses _pre_augnet_replicas suffix.")
    parser.add_argument("--random-init", action="store_true", default=False,
                        help="Use randomly initialized GNN + AugNet (no checkpoints). "
                             "Uses _random_init suffix. Mutually exclusive with --pre-augnet, "
                             "--replica-split, --use-fmap, --random-init-* flags.")
    parser.add_argument("--ood-query-dataset", type=str, default=None,
                        help="Legacy OOD: swap query embeddings to this dataset's graphs "
                             "(embedded by args.dataset's AugNet); corpus + oracle stay with "
                             "args.dataset. Requires --legacy-ood. Adds _ood_query=<ds> suffix.")
    parser.add_argument("--legacy-ood", action="store_true", default=False,
                        help="Enable the legacy OOD query-swap (requires --ood-query-dataset).")
    add_aug_args(parser)
    add_fmap_args(parser)
    args = parser.parse_args()

    if args.subtract_mean and args.trained_hyper:
        parser.error("--subtract-mean is not supported for trained hyperplanes")
    if args.pre_augnet and args.use_fmap:
        parser.error("--pre-augnet and --use-fmap are mutually exclusive")
    if args.pre_augnet and (args.random_init_gm_q or args.random_init_w0_q or args.random_init_gm_c):
        parser.error("--random-init-* flags are not supported with --pre-augnet")
    if args.use_fmap and args.trained_hyper:
        parser.error("--trained-hyper is not supported with --use-fmap")
    if args.use_fmap and (args.random_init_gm_q or args.random_init_w0_q or args.random_init_gm_c):
        parser.error("--random-init-* flags are not supported with --use-fmap")
    if args.random_init and (args.pre_augnet or args.replica_split or args.use_fmap
                             or args.random_init_gm_q or args.random_init_w0_q or args.random_init_gm_c):
        parser.error("--random-init is mutually exclusive with --pre-augnet, --replica-split, "
                     "--use-fmap, and --random-init-* flags")
    if args.ood_query_dataset and not args.legacy_ood:
        parser.error("--ood-query-dataset requires --legacy-ood (only legacy query-swap OOD is supported here)")
    if args.legacy_ood and not args.ood_query_dataset:
        parser.error("--legacy-ood requires --ood-query-dataset")
    if args.ood_query_dataset and (args.use_fmap or args.pre_augnet or args.replica_split
                                   or args.random_init):
        parser.error("--ood-query-dataset is not supported with --use-fmap, --pre-augnet, "
                     "--replica-split, or --random-init")

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    out_base = f"data/{args.dataset}/preprocessed/splits/train/clusters"

    if args.use_fmap:
        cluster_runner = build_fmap_runner(args)
        augnet_name = convert_name(cluster_runner.conf.task.name)
        precomputed_embs = compute_fmap_train_embeddings(cluster_runner)
    elif args.random_init:
        cluster_runner = build_random_init_runner(args)
        augnet_name = convert_name(cluster_runner.conf.task.name) + "_random_init"
        precomputed_embs = compute_train_embeddings(cluster_runner, pre_augnet=False)
    else:
        cluster_runner = build_runner(args)
        augnet_name = convert_name(cluster_runner.conf.task.name)
        if args.replica_split:
            augnet_name     += "_pre_augnet_replicas"
            precomputed_embs = compute_train_embeddings(cluster_runner, pre_augnet=True)
        elif args.pre_augnet:
            augnet_name     += "_pre_augnet"
            precomputed_embs = compute_train_embeddings(cluster_runner, pre_augnet=True)
        else:
            precomputed_embs = None

    name_suffix = f"_train_{args.third_loss_type}" if args.trained_hyper else ""
    query_graph_model = None
    corpus_graph_model = None
    override_W0_q = None

    if not args.use_fmap:
        seed = cluster_runner.conf.training.seed
        model_cls = f"{cluster_runner.conf.model.classPath}.{cluster_runner.conf.model.name}"
        device = cluster_runner.conf.training.device

        if args.random_init_gm_q:
            torch.manual_seed(seed)
            fresh_model = copy.deepcopy(cluster_runner.trainer.graph_model_best)
            for module in fresh_model.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            fresh_model.eval()
            query_graph_model = fresh_model
            name_suffix += "_rand_init_q"
            print(f"Using randomly initialized query GNN (seed={seed})")

        if args.random_init_w0_q:
            torch.manual_seed(seed)
            fresh_gm = get_class(model_cls)(
                cluster_runner.conf, cluster_runner.gmn_config
            ).to(device)
            override_W0_q = fresh_gm.last_linear_layer_params.detach()
            name_suffix += "_rand_init_w0_q"
            print(f"Using randomly initialized W_0 for q_hat (seed={seed})")

        if args.random_init_gm_c:
            torch.manual_seed(seed)
            fresh_model = copy.deepcopy(cluster_runner.trainer.graph_model)
            for module in fresh_model.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            fresh_model.eval()
            corpus_graph_model = fresh_model
            name_suffix += "_rand_init_c"
            print(f"Using randomly initialized corpus GNN for cosine scoring (seed={seed})")

        if not args.pre_augnet and not args.replica_split and not args.random_init:
            load_aug_checkpoint(cluster_runner)

    # Legacy OOD: swap query embeddings to ood_query_dataset (embedded by args.dataset's
    # AugNet); corpus + oracle (q_gts) stay with args.dataset. Mirrors gen_dataset_carefully.
    if args.ood_query_dataset:
        ood_runner = build_ood_query_runner(args, args.ood_query_dataset)
        load_aug_checkpoint(ood_runner)
        ood_query_aug_embs, _, _ = compute_train_embeddings(ood_runner)
        _, corpus_aug_embs, q_gts = compute_train_embeddings(cluster_runner)
        precomputed_embs = (ood_query_aug_embs, corpus_aug_embs, q_gts)
        name_suffix += f"_ood_query={args.ood_query_dataset}"
        print(f"OOD query embeddings: {tuple(ood_query_aug_embs.shape)} "
              f"from {args.ood_query_dataset}; corpus/oracle from {args.dataset}")

    if args.mode == "lsh":
        if args.overwrite:
            if args.trained_hyper:
                hp = select_trained_hyperplanes(
                    args.dataset, augnet_name,
                    k_over_c_min=args.k_over_c_min, k_over_c_max=args.k_over_c_max,
                    third_loss_type=args.third_loss_type,
                )
            else:
                hp = select_hyperplanes(
                    args.dataset, augnet_name,
                    k_over_c_min=args.k_over_c_min, k_over_c_max=args.k_over_c_max,
                )
        else:
            hp = None

        print(f"Running LSH noisy rerank for cluster {args.cluster}, sigmas={args.sigmas}, ratio={args.ratio}, overwrite={args.overwrite}")
        run_lsh_mode(
            hp, cluster_runner, cluster_runner.conf, args.ratio,
            args.cluster, args.sigmas, args.out_plots,
            out_base=out_base, dataset=args.dataset, augnet_name=augnet_name,
            K_val=args.K_val, overwrite=args.overwrite, name_suffix=name_suffix,
            query_graph_model=query_graph_model, corpus_graph_model=corpus_graph_model,
            override_W0_q=override_W0_q, precomputed_embs=precomputed_embs,
            replica_split=args.replica_split,
        )
    else:
        print(f"Plotting random relationships for cluster {args.cluster}, ratio={args.ratio}")
        plot_random_mode(
            cluster_runner, args.ratio, args.cluster, args.out_plots,
            out_base=out_base, dataset=args.dataset,
            K_val=args.K_val, name_suffix=name_suffix,
            query_graph_model=query_graph_model, corpus_graph_model=corpus_graph_model,
            override_W0_q=override_W0_q, precomputed_embs=precomputed_embs,
        )


if __name__ == "__main__":
    main()
