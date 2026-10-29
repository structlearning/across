"""
Generate LSH or random relationship files for a single cluster using the
carefully-maintained pos:neg ratio approach (mirrors generate_dataset_carefully
from attic/notebooks/get_rlsh_points.ipynb).

Usage
-----
# LSH mode, no ratio downsampling
python scripts/gen_dataset_carefully.py --dataset ptc_fr --cluster 3

# LSH mode, downsample to ratio=0.13
python scripts/gen_dataset_carefully.py --dataset ptc_fr --cluster 3 --ratio 0.13

# Random mode
python scripts/gen_dataset_carefully.py --dataset ptc_fr --cluster 3 --mode random
python scripts/gen_dataset_carefully.py --dataset ptc_fr --cluster 3 --mode random --ratio 0.13

# LSH flags
python scripts/gen_dataset_carefully.py --dataset ptc_fr --cluster 3 --rand_on_lsh   # random within LSH bucket
python scripts/gen_dataset_carefully.py --dataset ptc_fr --cluster 3 --bottom_k      # bottom-K instead of top-K
python scripts/gen_dataset_carefully.py --dataset ptc_fr --cluster 3 --ratio 0.13 --legacy_rng  # reproduce pre-fix files
"""

import argparse
import os
import sys
import pickle

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
from reform.utils.script_utils import (
    K_VAL,
    ratio_suffix as _ratio_suffix,
    add_aug_args, build_runner, build_random_init_runner, build_ood_query_runner, build_target_runner,
    build_aug_ckpt_tag_for, load_aug_checkpoint,
    add_fmap_args, build_fmap_runner, compute_fmap_train_embeddings,
    select_hyperplanes, select_trained_hyperplanes,
    sample_negatives, compute_train_embeddings,
    print_relationship_stats,
)


# ── constants ─────────────────────────────────────────────────────────────────
LSH_HP_SEED = 0


# ── LSH mode ──────────────────────────────────────────────────────────────────
def run_lsh_mode(
    hp, cluster_runner, conf, ratio, cluster_idx, K_val,
    out_base, augnet_name,
    rand_on_lsh=False, bottom_k=False, legacy_rng=False,
    ood_query_aug_embs=None, ood_query_dataset=None,
    ood_src_dataset=None,
    precomputed_embs=None,
    trained_suffix="",
    replica_split=False,
):
    if precomputed_embs is not None:
        query_aug_embs, corpus_aug_embs, q_gts = precomputed_embs
    else:
        query_aug_embs, corpus_aug_embs, q_gts = compute_train_embeddings(cluster_runner)
    if ood_query_aug_embs is not None:
        query_aug_embs = ood_query_aug_embs
    num_corpus  = corpus_aug_embs.shape[0]
    num_queries = query_aug_embs.shape[0]

    if replica_split:
        R = conf.aug_embedder.num_hyperplanes
        N_c, D, K = corpus_aug_embs.shape
        _embd_dim = K // R
        c_lsh = corpus_aug_embs.to("cpu").view(N_c, D, R, _embd_dim).view(N_c, D * R, _embd_dim).permute(1, 0, 2)
        lsh = LSH(torch.from_numpy(hp["hp_weights"]), torch.from_numpy(hp["hp_biases"]), hp["bit_indices"], c_lsh)
    else:
        lsh = LSH(
            torch.from_numpy(hp["hp_weights"]),
            torch.from_numpy(hp["hp_biases"]),
            hp["bit_indices"],
            corpus_aug_embs.to("cpu").permute(1, 0, 2),
        )

    c_normed    = F.normalize(corpus_aug_embs[:, 0, :], dim=-1)
    q_normed    = F.normalize(query_aug_embs[:,  0, :], dim=-1)
    cosine_sims = (q_normed @ c_normed.T).cpu()
    print(
        f"query_aug_embs: {query_aug_embs.shape}, "
        f"corpus_aug_embs: {corpus_aug_embs.shape}, "
        f"cosine_sims: {cosine_sims.shape}"
    )

    lsh_relationships = {}
    no_retr           = 0
    # mutable ref so sample_negatives can overwrite the rng (legacy mode only)
    rng_ref = [None]

    for local_idx, (query, scores, gt) in enumerate(zip(query_aug_embs, cosine_sims, q_gts)):
        q_lsh = query.cpu().view(D, R, _embd_dim).view(D * R, _embd_dim) if replica_split else query.cpu()
        c_ids        = lsh.retrieve(q_lsh)
        sorted_c_ids = sorted(c_ids)

        if rand_on_lsh:
            rng_ref[0]  = np.random.default_rng(conf.training.seed + local_idx)
            top_k_c_ids = rng_ref[0].choice(
                sorted_c_ids, size=min(K_val, len(sorted_c_ids)), replace=False
            ).tolist()
        elif bottom_k:
            top_k_idxs  = torch.argsort(scores[sorted_c_ids], descending=False)[:K_val]
            top_k_c_ids = [sorted_c_ids[i] for i in top_k_idxs.tolist()]
        else:
            top_k_idxs  = torch.argsort(scores[sorted_c_ids], descending=True)[:K_val]
            top_k_c_ids = [sorted_c_ids[i] for i in top_k_idxs.tolist()]

        pos_c_ids = [cid for cid in top_k_c_ids if gt[cid]]
        if not pos_c_ids:
            pos_c_ids = [0]
            print(f"Local idx {local_idx} has no positives in retrieved set; using dummy pos_c_id=0")
            no_retr += 1

        neg_c_ids = sorted(set(range(num_corpus)) - set(pos_c_ids))
        neg_c_ids = sample_negatives(
            neg_c_ids, ratio, pos_c_ids,
            seed=conf.training.seed + local_idx,
            legacy_rng_ref=rng_ref if legacy_rng else None,
        )

        if not neg_c_ids:
            print(f"Local idx {local_idx} has no negatives; removing 5 positives to make room")
            pos_c_ids = pos_c_ids[:-5]
            neg_c_ids = np.random.choice(
                list(set(range(num_corpus)) - set(pos_c_ids)),
                size=min(5, num_corpus - len(pos_c_ids)), replace=False,
            ).tolist()

        lsh_relationships[local_idx] = {"pos": pos_c_ids, "neg": neg_c_ids}

    print(f"LSH: {num_queries - no_retr}/{num_queries} queries have >=1 oracle positive")
    print_relationship_stats(lsh_relationships, "LSH")

    out_dir = f"{out_base}/{cluster_idx}/{augnet_name}"
    os.makedirs(out_dir, exist_ok=True)

    if ood_src_dataset:
        ood_sfx = f"_ood_src={ood_src_dataset}"
    elif ood_query_dataset:
        ood_sfx = f"_ood_query={ood_query_dataset}"
    else:
        ood_sfx = ""
    suffix = f"all_corpus_negs_K={K_val}{_ratio_suffix(ratio)}{ood_sfx}{trained_suffix}"
    with open(f"{out_dir}/lsh_relationships_{suffix}.pkl", "wb") as f:
        pickle.dump(lsh_relationships, f)

    metadata = {
        "hp_weights":        hp["hp_weights"],
        "hp_biases":         hp["hp_biases"],
        "bit_indices":       hp["bit_indices"],
        "K_val":             K_val,
        "aug_ckpt_tag":      cluster_runner.conf.task.name,
        "ratio":             ratio,
        "rand_on_lsh":       rand_on_lsh,
        "bottom_k":          bottom_k,
        "legacy_rng":        legacy_rng,
        "ood_query_dataset": ood_query_dataset,
        "ood_src_dataset":   ood_src_dataset,
    }
    with open(f"{out_dir}/lsh_metadata_{suffix}.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved to {out_dir}/")
    print(f"  lsh_relationships_{suffix}.pkl")
    print(f"  lsh_metadata_{suffix}.pkl")


# ── Random mode ───────────────────────────────────────────────────────────────
def run_random_mode(cluster_runner, conf, ratio, cluster_idx, K_val,
                    out_base, legacy_rng=False, precomputed_embs=None):
    if precomputed_embs is not None:
        query_aug_embs, corpus_aug_embs, q_gts = precomputed_embs
    else:
        query_aug_embs, corpus_aug_embs, q_gts = compute_train_embeddings(cluster_runner)
    num_corpus  = corpus_aug_embs.shape[0]
    num_queries = query_aug_embs.shape[0]

    # shared outer rng advances sequentially (mirrors notebook)
    rng                  = np.random.default_rng(conf.training.seed)
    rng_ref              = [rng]
    random_relationships = {}
    no_retr              = 0

    for local_idx, gt in enumerate(q_gts):
        random_c_ids = rng_ref[0].choice(num_corpus, size=K_val, replace=False).tolist()
        pos_c_ids    = [cid for cid in random_c_ids if gt[cid]]
        if not pos_c_ids:
            pos_c_ids = [0]
            no_retr  += 1

        neg_c_ids = sorted(set(range(num_corpus)) - set(pos_c_ids))
        neg_c_ids = sample_negatives(
            neg_c_ids, ratio, pos_c_ids,
            seed=conf.training.seed + local_idx,
            legacy_rng_ref=rng_ref if legacy_rng else None,
        )

        random_relationships[local_idx] = {"pos": pos_c_ids, "neg": neg_c_ids}

    print(f"Random: {num_queries - no_retr}/{num_queries} queries have >=1 oracle positive")
    print_relationship_stats(random_relationships, "random")

    out_dir = f"{out_base}/{cluster_idx}"
    assert os.path.exists(out_dir), f"Output directory {out_dir} does not exist."

    suffix = f"all_corpus_negs_K={K_val}{_ratio_suffix(ratio)}"
    with open(f"{out_dir}/random_relationships_{suffix}.pkl", "wb") as f:
        pickle.dump(random_relationships, f)

    metadata = {
        "K_val":        K_val,
        "aug_ckpt_tag": cluster_runner.conf.task.name,
        "ratio":        ratio,
        "legacy_rng":   legacy_rng,
    }
    with open(f"{out_dir}/random_metadata_{suffix}.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved to {out_dir}/")
    print(f"  random_relationships_{suffix}.pkl")
    print(f"  random_metadata_{suffix}.pkl")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate LSH or random relationship files for a cluster."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g. ptc_fr)")
    parser.add_argument("--cluster", type=int, required=True,
                        help="Cluster index (1-4)")
    parser.add_argument("--gpu", type=str, default=None,
                        help="CUDA_VISIBLE_DEVICES value (default: inherit from environment)")
    parser.add_argument("--mode", type=str, default="lsh", choices=["lsh", "random"],
                        help="Relationship mode: 'lsh' or 'random' (default: lsh)")
    parser.add_argument("--ratio", type=float, default=0.13,
                        help="Positive-to-negative ratio for neg downsampling. "
                             "If omitted, all remaining corpus items are used as negatives.")
    parser.add_argument("--K_val", type=int, default=K_VAL,
                        help=f"Number of corpus items to retrieve/sample per query (default: {K_VAL})")
    parser.add_argument("--trained-hyper", action="store_true", default=False,
                        help="[LSH only] Use trained LSH hyperplanes instead of random hyperplanes")
    parser.add_argument("--third-loss-type", type=str, default="sc_coll_min",
                        choices=["sc_coll_min", "ad_decorr"],
                        help="[LSH only, --trained-hyper] Third loss type used during LSH training (default: sc_coll_min)")
    # LSH-only flags
    parser.add_argument("--rand_on_lsh", action="store_true",
                        help="[LSH only] Sample K_val corpus IDs randomly from the LSH bucket "
                             "instead of selecting by top-K cosine similarity")
    parser.add_argument("--bottom_k", action="store_true",
                        help="[LSH only] Select bottom-K (lowest cosine sim) from LSH bucket "
                             "instead of top-K (picks hardest positives)")
    parser.add_argument("--legacy_rng", action="store_true",
                        help="Use pre-fix RNG behaviour (overwrites outer rng variable). "
                             "Set to reproduce files generated before the bug fix.")
    parser.add_argument("--ood-query-dataset", type=str, default=None,
                        help="[LSH only, AugNet only] OOD dataset. Default (new OOD): --dataset is "
                             "the target; AugNet + hyperplanes come from --ood-query-dataset (src). "
                             "With --legacy-ood: --dataset is src corpus; queries come from this dataset.")
    parser.add_argument("--legacy-ood", action="store_true", default=False,
                        help="[LSH only] Use the legacy OOD scheme: src corpus + OOD queries "
                             "(requires --ood-query-dataset). Default is the new OOD scheme: "
                             "target corpus/queries + src AugNet/hyperplanes.")
    parser.add_argument("--use-fmap", action="store_true", default=False,
                        help="Use FMap embeddings instead of AugNet embeddings")
    parser.add_argument("--pre-augnet", action="store_true", default=False,
                        help="Use pre-AugNet (random-feature-map) embeddings instead of AugNet MLP outputs")
    parser.add_argument("--replica-split", action="store_true", default=False,
                        help="Implies --pre-augnet; each hyperplane gets its own hash table. "
                             "Uses _pre_augnet_replicas suffix.")
    parser.add_argument("--random-init", action="store_true", default=False,
                        help="Use randomly initialized GNN + AugNet (no checkpoints). "
                             "Uses _random_init suffix. Mutually exclusive with --ood-query-dataset, "
                             "--use-fmap, --pre-augnet, --replica-split.")
    add_aug_args(parser)
    add_fmap_args(parser)
    args = parser.parse_args()

    if args.rand_on_lsh and args.bottom_k:
        parser.error("--rand_on_lsh and --bottom_k are mutually exclusive")
    if args.mode == "random" and (args.rand_on_lsh or args.bottom_k):
        parser.error("--rand_on_lsh and --bottom_k are only valid with --mode lsh")
    if args.mode == "random" and args.ood_query_dataset:
        parser.error("--ood-query-dataset is only valid with --mode lsh")
    if args.use_fmap and args.ood_query_dataset:
        parser.error("--ood-query-dataset is not supported with --use-fmap")
    if args.pre_augnet and args.ood_query_dataset:
        parser.error("--ood-query-dataset is not supported with --pre-augnet")
    if args.replica_split and args.ood_query_dataset:
        parser.error("--ood-query-dataset is not supported with --replica-split")
    if args.replica_split and args.use_fmap:
        parser.error("--replica-split and --use-fmap are mutually exclusive")
    if args.pre_augnet and args.use_fmap:
        parser.error("--pre-augnet and --use-fmap are mutually exclusive")
    if args.use_fmap and args.trained_hyper:
        parser.error("--trained-hyper is not supported with --use-fmap")
    if args.legacy_ood and not args.ood_query_dataset:
        parser.error("--legacy-ood requires --ood-query-dataset")
    if args.random_init and (args.ood_query_dataset or args.use_fmap
                             or args.pre_augnet or args.replica_split):
        parser.error("--random-init is mutually exclusive with --ood-query-dataset, "
                     "--use-fmap, --pre-augnet, and --replica-split")

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    out_base = f"data/{args.dataset}/preprocessed/splits/train/clusters"

    ood_query_aug_embs = None
    ood_src_dataset    = None

    if args.random_init:
        cluster_runner   = build_random_init_runner(args)
        augnet_name      = convert_name(cluster_runner.conf.task.name) + "_random_init"
        precomputed_embs = compute_train_embeddings(cluster_runner, pre_augnet=False)
    elif args.use_fmap:
        cluster_runner   = build_fmap_runner(args)
        augnet_name      = convert_name(cluster_runner.conf.task.name)
        precomputed_embs = compute_fmap_train_embeddings(cluster_runner)
    elif args.ood_query_dataset and not args.legacy_ood:
        # New OOD: target corpus+queries (args.dataset), src AugNet+hyperplanes (ood_query_dataset)
        src_dataset      = args.ood_query_dataset
        cluster_runner   = build_target_runner(args, src_dataset)
        augnet_name      = convert_name(build_aug_ckpt_tag_for(src_dataset, args))
        load_aug_checkpoint(cluster_runner)
        precomputed_embs = None
        ood_src_dataset  = src_dataset
    else:
        cluster_runner   = build_runner(args)
        augnet_name      = convert_name(cluster_runner.conf.task.name)
        if args.replica_split:
            augnet_name     += "_pre_augnet_replicas"
            precomputed_embs = compute_train_embeddings(cluster_runner, pre_augnet=True)
        elif args.pre_augnet:
            augnet_name     += "_pre_augnet"
            precomputed_embs = compute_train_embeddings(cluster_runner, pre_augnet=True)
        else:
            load_aug_checkpoint(cluster_runner)
            precomputed_embs = None
        if args.ood_query_dataset:  # legacy OOD: swap queries only
            ood_runner = build_ood_query_runner(args, args.ood_query_dataset)
            load_aug_checkpoint(ood_runner)
            ood_query_aug_embs, _, _ = compute_train_embeddings(ood_runner)
            print(f"OOD query embeddings: {ood_query_aug_embs.shape} from {args.ood_query_dataset}")

    if args.mode == "lsh":
        # Hyperplanes come from src dataset in new OOD mode, own dataset otherwise
        hp_dataset = ood_src_dataset if ood_src_dataset else args.dataset
        if args.trained_hyper:
            hp = select_trained_hyperplanes(
                hp_dataset, augnet_name,
                k_over_c_min=args.k_over_c_min, k_over_c_max=args.k_over_c_max,
                third_loss_type=args.third_loss_type,
            )
        else:
            hp = select_hyperplanes(
                hp_dataset, augnet_name,
                k_over_c_min=args.k_over_c_min, k_over_c_max=args.k_over_c_max,
            )

        print(
            f"\nRunning LSH mode: cluster={args.cluster}, K_val={args.K_val}, "
            f"ratio={args.ratio}, rand_on_lsh={args.rand_on_lsh}, "
            f"bottom_k={args.bottom_k}, legacy_rng={args.legacy_rng}"
        )
        trained_suffix = f"_train_{args.third_loss_type}" if args.trained_hyper else ""
        run_lsh_mode(
            hp, cluster_runner, cluster_runner.conf, args.ratio,
            args.cluster, args.K_val,
            out_base=out_base, augnet_name=augnet_name,
            rand_on_lsh=args.rand_on_lsh,
            bottom_k=args.bottom_k,
            legacy_rng=args.legacy_rng,
            ood_query_aug_embs=ood_query_aug_embs,
            ood_query_dataset=args.ood_query_dataset if args.legacy_ood else None,
            ood_src_dataset=ood_src_dataset,
            precomputed_embs=precomputed_embs,
            trained_suffix=trained_suffix,
            replica_split=args.replica_split,
        )
    else:
        print(
            f"\nRunning random mode: cluster={args.cluster}, K_val={args.K_val}, "
            f"ratio={args.ratio}, legacy_rng={args.legacy_rng}"
        )
        run_random_mode(
            cluster_runner, cluster_runner.conf, args.ratio,
            args.cluster, args.K_val,
            out_base=out_base,
            legacy_rng=args.legacy_rng,
            precomputed_embs=precomputed_embs,
        )


if __name__ == "__main__":
    main()
