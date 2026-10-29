"""
Shared utilities for active-selection and augnet scripts
(e.g. gen_dataset_carefully.py, noisy_rerank_plot.py, gen_gnn_distance.py).
"""

import argparse
import glob
import pickle
import re
from pathlib import Path

import numpy as np
import torch

from reform.runner.aug_emb_run import AugEmbRunner


CORPUS_SIZE = 100_000
K_VAL       = 40


def ratio_suffix(ratio):
    return f"_ratio={ratio}" if ratio is not None else ""


def _fmt_sigma(v: float) -> str:
    return str(int(v)) if v == int(v) else str(v)


def noise_dim_suffix(sigma_query: float, sigma_corpus: float) -> str:
    if sigma_query == 0.0 and sigma_corpus == 0.0:
        return ""
    return f"_noise_dim_sq={_fmt_sigma(sigma_query)}_sc={_fmt_sigma(sigma_corpus)}"


def sample_negatives(neg_c_ids, ratio, pos_c_ids, seed, legacy_rng_ref=None):
    """Downsample negatives to maintain pos:neg == ratio, or return all if ratio is None.

    legacy_rng_ref is a mutable list [rng] used to reproduce the original buggy
    rng-overwrite behaviour in gen_dataset_carefully.py; pass None for normal use.
    """
    if ratio is None:
        return neg_c_ids
    desired = int(len(pos_c_ids) / ratio)
    if len(neg_c_ids) > desired:
        if legacy_rng_ref is not None:
            legacy_rng_ref[0] = np.random.default_rng(seed)
            neg_c_ids = legacy_rng_ref[0].choice(neg_c_ids, size=desired, replace=False).tolist()
        else:
            neg_rng   = np.random.default_rng(seed)
            neg_c_ids = neg_rng.choice(neg_c_ids, size=desired, replace=False).tolist()
    return neg_c_ids


def print_relationship_stats(relationships, label):
    total_pos = sum(len(v["pos"]) for v in relationships.values())
    pos_neg_ratios = [
        len(v["pos"]) / len(v["neg"])
        for v in relationships.values() if len(v["neg"]) > 0
    ]
    print(f"Total number of pos ({label}): {total_pos}")
    print(f"Average positive:negative ratio across queries: {np.mean(pos_neg_ratios):.4f}")


def build_relationships_from_top_k(q_gts, top_k_per_query, num_corpus, ratio, base_seed):
    """Build a relationships dict from pre-computed top-K corpus IDs per query.

    Probes the oracle (q_gts) for positives, falls back to dummy pos_c_id=0 when
    none are found, and downsamples negatives to maintain the pos:neg ratio.

    Returns (relationships, no_pos_count).
    """
    relationships = {}
    no_pos_count  = 0
    for local_idx, (gt, top_k_c_ids) in enumerate(zip(q_gts, top_k_per_query)):
        pos_c_ids = [cid for cid in top_k_c_ids if gt[cid]]
        if not pos_c_ids:
            pos_c_ids = [0]
            print(f"Local idx {local_idx} has no positives in retrieved set; using dummy pos_c_id=0")
            no_pos_count += 1

        neg_c_ids = sorted(set(range(num_corpus)) - set(pos_c_ids))
        neg_c_ids = sample_negatives(neg_c_ids, ratio, pos_c_ids, seed=base_seed + local_idx)

        if not neg_c_ids:
            print(f"Local idx {local_idx} has no negatives; removing 5 positives to make room")
            pos_c_ids = pos_c_ids[:-5]
            neg_c_ids = np.random.choice(
                list(set(range(num_corpus)) - set(pos_c_ids)),
                size=min(5, num_corpus - len(pos_c_ids)), replace=False,
            ).tolist()

        relationships[local_idx] = {"pos": pos_c_ids, "neg": neg_c_ids}
    return relationships, no_pos_count


def compute_train_embeddings(cluster_runner, query_graph_model=None,
                             corpus_graph_model=None, override_W0_q=None,
                             pre_augnet=False):
    """Load train-split embeddings and ground truth for a cluster.

    When pre_augnet=True, returns raw AugmentedEmbedder output (N, D, K)
    without applying the AugNet MLP.
    """
    trainer  = cluster_runner.trainer
    train_ds = trainer.train_dataset
    query_aug_embs, corpus_aug_embs, _ = trainer._validate_common(
        which="train", preserve_dim=True,
        query_graph_model=query_graph_model,
        corpus_graph_model=corpus_graph_model,
        override_W0_q=override_W0_q,
        pre_augnet=pre_augnet,
    )
    q_gts = train_ds.all_gt.reshape(
        query_aug_embs.shape[0], corpus_aug_embs.shape[0]
    )
    return query_aug_embs, corpus_aug_embs, q_gts


def cosine_sims_for_relationships(relationships, cosine_sims, q_gts):
    """Extract cosine sim values for pos/neg pairs and count queries with a real oracle positive."""
    pos_vals, neg_vals = [], []
    queries_with_pos   = 0
    for local_idx, rel in relationships.items():
        gt = q_gts[local_idx]
        if any(gt[cid] for cid in rel["pos"]):
            queries_with_pos += 1
        for cid in rel["pos"]:
            pos_vals.append(cosine_sims[local_idx][cid].item())
        for cid in rel["neg"]:
            neg_vals.append(cosine_sims[local_idx][cid].item())
    return pos_vals, neg_vals, queries_with_pos


def add_aug_args(parser: argparse.ArgumentParser) -> None:
    """Add augnet architecture arguments to an argparse parser."""
    g = parser.add_argument_group("augnet architecture")
    g.add_argument("--aug-hidden-dim", type=int, default=64,
                   help="AugNet MLP hidden dimension (default: 64)")
    g.add_argument("--aug-num-layers", type=int, default=2,
                   help="AugNet MLP number of layers (default: 2)")
    g.add_argument("--aug-output-dim", type=int, default=10,
                   help="AugNet output dimension (default: 10)")
    g.add_argument("--aug-form", type=str, default="short", choices=["short", "long"],
                   help="Augmented vector form passed to the augnet (default: short)")
    g.add_argument("--aug-model-every-dim", action=argparse.BooleanOptionalAction, default=False,
                   help="Use a separate MLP per GNN output dimension (default: False)")
    g.add_argument("--aug-push-minus", action=argparse.BooleanOptionalAction, default=True,
                   help="Negate query input to the augnet (default: True)")
    g.add_argument("--aug-symmetric", action=argparse.BooleanOptionalAction, default=True,
                   help="Share query/corpus weights in the augnet (default: True)")
    g.add_argument("--aug-loss", type=str, default="ce-adv-max",
                   help="Loss used during augnet training, for checkpoint tag (default: ce-adv-max)")
    g.add_argument("--estol", type=float, default=5e-3,
                   help="Early stopping tolerance used during augnet training, for checkpoint tag (default: 5e-3)")
    g.add_argument("--pgd-epsilon", type=float, default=0.1,
                   help="PGD ball radius used during ce-adv-max augnet training, for checkpoint tag (default: 0.1)")
    g.add_argument("--num-epochs", type=int, default=None,
                   help="Override training.num_epochs (default: use config value)")
    g = parser.add_argument_group("LSH hyperplane selection")
    g.add_argument("--k-over-c-min", type=float, default=0.2,
                   help="Lower bound of k/C range for hyperplane efficiency selection (default: 0.2)")
    g.add_argument("--k-over-c-max", type=float, default=0.6,
                   help="Upper bound of k/C range for hyperplane efficiency selection (default: 0.6)")


def _build_ckpt_tag(args) -> str:
    """GNN checkpoint tag, mirroring RunnerBase.set_task_name()."""
    return (
        f"GEN_{args.dataset}_numC={CORPUS_SIZE}_MinR=0.05_MaxR=0.25"
        "_scoring_function=cos,stemp=0.1,margin=1.0"
    )


def build_aug_ckpt_tag_for(dataset: str, args) -> str:
    """AugNet checkpoint tag for an arbitrary dataset with architecture from args."""
    base = (
        f"GEN_{dataset}_numC={CORPUS_SIZE}_MinR=0.05_MaxR=0.25"
        "_scoring_function=cos,stemp=0.1,margin=1.0"
    )
    if args.estol != 5e-3:
        base += f"_estol={args.estol}"
    sym_str = "sym" if args.aug_symmetric else "asym"
    return (
        f"{base}_aug_emb_mlp_{sym_str}"
        f"_h{args.aug_hidden_dim}_l{args.aug_num_layers}_o{args.aug_output_dim}"
        f"_loss{args.aug_loss}_af{args.aug_form}"
        + ("_everydim" if args.aug_model_every_dim else "")
        + ("_pushminus" if args.aug_push_minus else "")
        + "_aug_corpus_initfix"
        + "_pretrained_embs"
        + (f"_eps={args.pgd_epsilon}" if args.pgd_epsilon != 0.1 else "")
    )


def _build_aug_ckpt_tag(args) -> str:
    """Augnet checkpoint tag, mirroring AugEmbRunner.set_task_name().

    Needed before runner construction to form the lsh.aug_checkpoint_path
    override; after construction cluster_runner.conf.task.name equals this.
    """
    return build_aug_ckpt_tag_for(args.dataset, args)


def _build_fmap_ckpt_tag(args) -> str:
    """FMap checkpoint tag, mirroring FMapRunner.set_task_name()."""
    ckpt_tag = _build_ckpt_tag(args)
    if args.estol != 5e-3:
        ckpt_tag += f"_estol={args.estol}"
    sym_str = "sym" if args.fmap_symmetric else "asym"
    return (
        f"{ckpt_tag}_fmap_{args.fmap_model}_{sym_str}"
        f"_h{args.fmap_hidden_dim}_l{args.fmap_num_layers}_o{args.fmap_output_dim}"
        f"_m{args.fmap_m_use}_T{args.fmap_T}"
        f"_loss{args.fmap_loss}"
        + ("_flat" if args.fmap_flat_mlp else "")
    )


def add_fmap_args(parser: argparse.ArgumentParser) -> None:
    """Add FMap architecture arguments to an argparse parser."""
    g = parser.add_argument_group("fmap architecture")
    g.add_argument("--fmap-model", type=str, default="mlp", choices=["mlp", "linear"])
    g.add_argument("--fmap-symmetric", action=argparse.BooleanOptionalAction, default=False)
    g.add_argument("--fmap-hidden-dim", type=int, default=64)
    g.add_argument("--fmap-num-layers", type=int, default=2)
    g.add_argument("--fmap-output-dim", type=int, default=10)
    g.add_argument("--fmap-m-use", type=int, default=10)
    g.add_argument("--fmap-T", type=float, default=3.0)
    g.add_argument("--fmap-loss", type=str, default="ce-avg",
                   help="Loss used during fmap training, for checkpoint tag (default: ce-avg)")
    g.add_argument("--fmap-flat-mlp", action=argparse.BooleanOptionalAction, default=False)
    try:
        g.add_argument("--estol", type=float, default=5e-3,
                       help="Early stopping tolerance used during training (default: 5e-3)")
    except argparse.ArgumentError:
        pass  # already added by add_aug_args


def build_runner(args) -> AugEmbRunner:
    """Construct and initialise an AugEmbRunner from parsed CLI args."""
    ckpt_tag     = _build_ckpt_tag(args)
    aug_ckpt_tag = _build_aug_ckpt_tag(args)

    print(f"\nSetting up runner for cluster {args.cluster} ...")
    cluster_runner = AugEmbRunner(overrides=[
        f"dataset.name={args.dataset}",
        "dataset.rel_mode=sub_iso",
        "model.name=GEN",
        "model.scoring_function=cos",
        f"dataset.aug_num_cgraphs={CORPUS_SIZE}",
        "dataset.MinR=0.05",
        "dataset.MaxR=0.25",
        "training.sinkhorn_temp=0.1",
        "training.margin=1.0",
        f"training.loss={args.aug_loss}",
        "training.batch_size=8192",
        "aug_embedder.model=mlp",
        f"aug_embedder.symmetric={args.aug_symmetric}",
        f"aug_embedder.hidden_dim={args.aug_hidden_dim}",
        f"aug_embedder.num_layers={args.aug_num_layers}",
        f"aug_embedder.output_dim={args.aug_output_dim}",
        f"aug_embedder.aug_form={args.aug_form}",
        f"aug_embedder.model_every_dim={args.aug_model_every_dim}",
        f"aug_embedder.push_minus={args.aug_push_minus}",
        f"aug_embedder.pgd_epsilon={args.pgd_epsilon}",
        # e.g. bestValidationModels/GEN_ptc_fr_numC=100000_MinR=0.05_MaxR=0.25_scoring_function=cos,stemp=0.1,margin=1.0/best.pt
        f"aug_embedder.best_checkpoint_path=bestValidationModels/{ckpt_tag}/best.pt",
        # e.g. latestModels/GEN_ptc_fr_numC=100000_MinR=0.05_MaxR=0.25_scoring_function=cos,stemp=0.1,margin=1.0
        f"aug_embedder.latest_ckpts_path=latestModels/{ckpt_tag}",
        # e.g. bestValidationModels/GEN_ptc_fr_numC=100000_MinR=0.05_MaxR=0.25_scoring_function=cos,stemp=0.1,margin=1.0_aug_emb_mlp_sym_h64_l2_o10_lossce-adv-max_afshort_everydim_pushminus_aug_corpus_initfix_pretrained_embs/best.pt
        f"lsh.aug_checkpoint_path=bestValidationModels/{aug_ckpt_tag}/best.pt",
        "lsh.dump_dir=./allPklDumps",
        "task.wandb_enabled=false",
        "aug_embedder.use_pretrained_embs=True",
        "dataset.data_type=idx",
        f"use_cluster={args.cluster}",
        "training.patience=50",
        f"early_stopping.tolerance={args.estol}",
    ])
    cluster_runner.set_trainer()
    return cluster_runner


def select_hyperplanes(
    dataset: str,
    augnet_name: str,
    k_over_c_min: float = 0.2,
    k_over_c_max: float = 0.6,
    lsh_key: tuple = (0.05, False, "LSH"),
    noise_sigma_query: float = 0.0,
    noise_sigma_corpus: float = 0.0,
) -> dict:
    """Search all sweep result files for the (seed, subset_size) with the
    highest efficiency (set_rec / k_over_C) in [k_over_c_min, k_over_c_max],
    then load and return the corresponding hyperplane dict.

    noise_sigma_query / noise_sigma_corpus select the noise-dim sweep files
    saved by lsh_retr_sweep.py --noise-sigma-query / --noise-sigma-corpus.
    """
    nd_sfx = noise_dim_suffix(noise_sigma_query, noise_sigma_corpus)
    sweep_pattern = (
        f"allPklDumps/{dataset}/"
        f"lsh_retr_sweep_{dataset}_{augnet_name}_cos_random_hyper_seed*{nd_sfx}.pkl"
    )
    sweep_files = glob.glob(sweep_pattern)
    if not sweep_files:
        raise FileNotFoundError(f"No sweep result files found matching: {sweep_pattern}")

    seed_re = re.compile(r"_seed(\d+)" + re.escape(nd_sfx) + r"\.pkl$")
    best_eff, best_seed, best_ss = -1.0, None, None

    for sweep_path in sweep_files:
        m = seed_re.search(sweep_path)
        if not m:
            continue
        seed = int(m.group(1))
        res_dict = pickle.load(open(sweep_path, "rb"))
        if lsh_key not in res_dict:
            continue
        for agg in res_dict[lsh_key]:
            k_over_C = agg["num_evals"] / CORPUS_SIZE
            if k_over_c_min <= k_over_C <= k_over_c_max:
                eff = agg["set_rec"] / k_over_C
                if eff > best_eff:
                    best_eff, best_seed, best_ss = eff, seed, agg["subset_size"]

    if best_seed is None:
        raise ValueError(
            f"No hyperplane candidates found in k/C ∈ [{k_over_c_min}, {k_over_c_max}] "
            f"across {len(sweep_files)} sweep file(s) matching: {sweep_pattern}"
        )

    hp_path = (
        f"allPklDumps/{dataset}/"
        f"lsh_hyperplanes_{dataset}_{augnet_name}_cos_random_hyper_seed{best_seed}{nd_sfx}.pkl"
    )
    rlsh_dict = pickle.load(open(hp_path, "rb"))
    hp = rlsh_dict[lsh_key][best_ss]
    print(f"Selected hyperplanes: seed={best_seed}, subset_size={best_ss}, efficiency={best_eff:.4f}")
    return hp


def select_trained_hyperplanes(
    dataset: str,
    augnet_name: str,
    k_over_c_min: float = 0.2,
    k_over_c_max: float = 0.6,
    third_loss_type: str = 'sc_coll_min',
) -> dict:
    """Search the trained-LSH sweep file for the (λ, linperd, subset_size) with
    highest efficiency (set_rec / k_over_C) in [k_over_c_min, k_over_c_max],
    then load and return the corresponding hyperplane dict.

    Returns {hp_weights, hp_biases, bit_indices} — same shape as select_hyperplanes.
    """
    loss_suffix = f"_{third_loss_type}" if third_loss_type != 'sc_coll_min' else ""
    base = f"allPklDumps/{dataset}"
    sweep_path = f"{base}/lsh_retr_sweep_{dataset}_{augnet_name}_cos{loss_suffix}.pkl"
    hp_path    = f"{base}/lsh_hyperplanes_{dataset}_{augnet_name}_cos{loss_suffix}.pkl"

    if not Path(sweep_path).exists():
        raise FileNotFoundError(f"No trained sweep file found: {sweep_path}")

    res_dict = pickle.load(open(sweep_path, "rb"))

    best_eff, best_key, best_ss = -1.0, None, None
    for (lam, linperd, wtype), results in res_dict.items():
        if wtype != "LSH" or results is None:
            continue
        for agg in results:
            k_over_C = agg["num_evals"] / CORPUS_SIZE
            if k_over_c_min <= k_over_C <= k_over_c_max:
                eff = agg["set_rec"] / k_over_C
                if eff > best_eff:
                    best_eff = eff
                    best_key = (lam, linperd, wtype)
                    best_ss  = agg["subset_size"]

    if best_key is None:
        raise ValueError(
            f"No trained hyperplane candidates found in k/C ∈ [{k_over_c_min}, {k_over_c_max}] "
            f"in {sweep_path}"
        )

    lam, linperd, _ = best_key
    print(f"Selected trained hyperplanes: λ={lam}, linperd={linperd}, subset_size={best_ss}, efficiency={best_eff:.4f}")

    hp_dict  = pickle.load(open(hp_path, "rb"))
    trained  = hp_dict[best_key]
    return {
        "hp_weights": trained["hp_weights"],
        "hp_biases":  trained["hp_biases"],
        "bit_indices": trained["bit_indices_by_subset"][best_ss],
    }


def build_fmap_runner(args):
    """Construct and initialise an FMapEmbRunner from parsed CLI args."""
    from reform.runner.fmap_run import FMapEmbRunner

    ckpt_tag      = _build_ckpt_tag(args)
    fmap_ckpt_tag = _build_fmap_ckpt_tag(args)

    print(f"\nSetting up FMap runner for cluster {args.cluster} ...")
    runner = FMapEmbRunner(overrides=[
        f"dataset.name={args.dataset}",
        "dataset.rel_mode=sub_iso",
        "model.name=GEN",
        "model.scoring_function=cos",
        f"dataset.aug_num_cgraphs={CORPUS_SIZE}",
        "dataset.MinR=0.05",
        "dataset.MaxR=0.25",
        "training.sinkhorn_temp=0.1",
        "training.margin=1.0",
        f"training.loss={args.fmap_loss}",
        "training.batch_size=8192",
        "fmap_embedder.model=mlp",
        f"fmap_embedder.symmetric={args.fmap_symmetric}",
        f"fmap_embedder.hidden_dim={args.fmap_hidden_dim}",
        f"fmap_embedder.num_layers={args.fmap_num_layers}",
        f"fmap_embedder.output_dim={args.fmap_output_dim}",
        f"fmap_embedder.m_use={args.fmap_m_use}",
        f"fmap_embedder.T={args.fmap_T}",
        f"fmap_embedder.flat_mlp={args.fmap_flat_mlp}",
        f"fmap_embedder.best_checkpoint_path=bestValidationModels/{ckpt_tag}/best.pt",
        f"fmap_embedder.latest_ckpts_path=latestModels/{ckpt_tag}",
        f"lsh.fmap_checkpoint_path=bestValidationModels/{fmap_ckpt_tag}/best.pt",
        "lsh.dump_dir=./allPklDumps",
        "task.wandb_enabled=false",
        "dataset.data_type=idx",
        f"use_cluster={args.cluster}",
        "training.patience=50",
        f"early_stopping.tolerance={args.estol}",
    ])
    runner.set_trainer()
    return runner


def compute_fmap_train_embeddings(fmap_runner):
    """Compute FMap train-split embeddings and ground truth for a cluster."""
    trainer  = fmap_runner.trainer
    train_ds = trainer.train_dataset
    query_embs, corpus_embs, _ = trainer._validate_common("train", preserve_dim=True)
    q_gts = train_ds.all_gt.reshape(query_embs.shape[0], corpus_embs.shape[0])
    return query_embs, corpus_embs, q_gts


def build_ood_query_runner(args, ood_dataset: str) -> AugEmbRunner:
    """Construct a runner that loads ood_dataset's graphs/pretrained embeddings
    but applies the src-dataset-trained AugNet (from args.dataset's checkpoint).

    Used for OOD query experiments: query_aug_embs come from ood_dataset graphs
    embedded by the src AugNet (applied on top of ood_dataset's own pretrained GNN
    embeddings), while corpus and oracle stay with args.dataset.
    """
    # GNN checkpoint must match the pretrained embeddings, which are keyed by ood_dataset
    ood_ckpt_tag     = (
        f"GEN_{ood_dataset}_numC={CORPUS_SIZE}_MinR=0.05_MaxR=0.25"
        "_scoring_function=cos,stemp=0.1,margin=1.0"
    )
    src_aug_ckpt_tag = _build_aug_ckpt_tag(args)

    print(f"\nSetting up OOD query runner "
          f"(queries={ood_dataset}, AugNet={args.dataset}, cluster={args.cluster}) ...")
    ood_runner = AugEmbRunner(overrides=[
        f"dataset.name={ood_dataset}",
        "dataset.rel_mode=sub_iso",
        "model.name=GEN",
        "model.scoring_function=cos",
        f"dataset.aug_num_cgraphs={CORPUS_SIZE}",
        "dataset.MinR=0.05",
        "dataset.MaxR=0.25",
        "training.sinkhorn_temp=0.1",
        "training.margin=1.0",
        f"training.loss={args.aug_loss}",
        "training.batch_size=8192",
        "aug_embedder.model=mlp",
        f"aug_embedder.symmetric={args.aug_symmetric}",
        f"aug_embedder.hidden_dim={args.aug_hidden_dim}",
        f"aug_embedder.num_layers={args.aug_num_layers}",
        f"aug_embedder.output_dim={args.aug_output_dim}",
        f"aug_embedder.aug_form={args.aug_form}",
        f"aug_embedder.model_every_dim={args.aug_model_every_dim}",
        f"aug_embedder.push_minus={args.aug_push_minus}",
        # ood GNN checkpoint — must match the pretrained embs at allPklDumps/{ood_dataset}/
        f"aug_embedder.best_checkpoint_path=bestValidationModels/{ood_ckpt_tag}/best.pt",
        f"aug_embedder.latest_ckpts_path=latestModels/{ood_ckpt_tag}",
        # src AugNet checkpoint — this is the key OOD transfer
        f"lsh.aug_checkpoint_path=bestValidationModels/{src_aug_ckpt_tag}/best.pt",
        "lsh.dump_dir=./allPklDumps",
        "task.wandb_enabled=false",
        "aug_embedder.use_pretrained_embs=True",
        "dataset.data_type=idx",
        f"use_cluster={args.cluster}",
        "training.patience=50",
        f"early_stopping.tolerance={args.estol}",
    ])
    ood_runner.set_trainer()
    return ood_runner


def build_target_runner(args, src_dataset: str) -> AugEmbRunner:
    """Construct a runner that loads target-dataset (args.dataset) graphs and pretrained
    embeddings, but applies the src-dataset-trained AugNet.

    New OOD scheme: corpus and queries come from args.dataset (target), but the AugNet
    checkpoint and hyperplanes are borrowed from src_dataset.
    """
    target_ckpt_tag  = (
        f"GEN_{args.dataset}_numC={CORPUS_SIZE}_MinR=0.05_MaxR=0.25"
        "_scoring_function=cos,stemp=0.1,margin=1.0"
    )
    src_aug_ckpt_tag = build_aug_ckpt_tag_for(src_dataset, args)

    print(f"\nSetting up target runner "
          f"(target={args.dataset}, AugNet src={src_dataset}, cluster={args.cluster}) ...")
    runner = AugEmbRunner(overrides=[
        f"dataset.name={args.dataset}",
        "dataset.rel_mode=sub_iso",
        "model.name=GEN",
        "model.scoring_function=cos",
        f"dataset.aug_num_cgraphs={CORPUS_SIZE}",
        "dataset.MinR=0.05",
        "dataset.MaxR=0.25",
        "training.sinkhorn_temp=0.1",
        "training.margin=1.0",
        f"training.loss={args.aug_loss}",
        "training.batch_size=8192",
        "aug_embedder.model=mlp",
        f"aug_embedder.symmetric={args.aug_symmetric}",
        f"aug_embedder.hidden_dim={args.aug_hidden_dim}",
        f"aug_embedder.num_layers={args.aug_num_layers}",
        f"aug_embedder.output_dim={args.aug_output_dim}",
        f"aug_embedder.aug_form={args.aug_form}",
        f"aug_embedder.model_every_dim={args.aug_model_every_dim}",
        f"aug_embedder.push_minus={args.aug_push_minus}",
        # target GNN checkpoint — matches pretrained embs at allPklDumps/{args.dataset}/
        f"aug_embedder.best_checkpoint_path=bestValidationModels/{target_ckpt_tag}/best.pt",
        f"aug_embedder.latest_ckpts_path=latestModels/{target_ckpt_tag}",
        # src AugNet checkpoint — the OOD transfer
        f"lsh.aug_checkpoint_path=bestValidationModels/{src_aug_ckpt_tag}/best.pt",
        "lsh.dump_dir=./allPklDumps",
        "task.wandb_enabled=false",
        "aug_embedder.use_pretrained_embs=True",
        "dataset.data_type=idx",
        f"use_cluster={args.cluster}",
        "training.patience=50",
        f"early_stopping.tolerance={args.estol}",
    ])
    runner.set_trainer()
    return runner


def build_random_init_runner(args) -> AugEmbRunner:
    """Like build_runner but GNN + AugNet both randomly initialized — no checkpoints loaded."""
    from reform.models import AugEmbeddingMap
    from reform.training.aug_emb_train import AugEmbTrainer
    from reform.utils.utils import get_class

    runner = AugEmbRunner(overrides=[
        f"dataset.name={args.dataset}",
        "dataset.rel_mode=sub_iso",
        "model.name=GEN",
        "model.scoring_function=cos",
        f"dataset.aug_num_cgraphs={CORPUS_SIZE}",
        "dataset.MinR=0.05",
        "dataset.MaxR=0.25",
        "training.sinkhorn_temp=0.1",
        "training.margin=1.0",
        f"training.loss={args.aug_loss}",
        "training.batch_size=8192",
        "aug_embedder.model=mlp",
        f"aug_embedder.symmetric={args.aug_symmetric}",
        f"aug_embedder.hidden_dim={args.aug_hidden_dim}",
        f"aug_embedder.num_layers={args.aug_num_layers}",
        f"aug_embedder.output_dim={args.aug_output_dim}",
        f"aug_embedder.aug_form={args.aug_form}",
        f"aug_embedder.model_every_dim={args.aug_model_every_dim}",
        f"aug_embedder.push_minus={args.aug_push_minus}",
        "lsh.dump_dir=./allPklDumps",
        "task.wandb_enabled=false",
        "aug_embedder.use_pretrained_embs=False",
        "aug_embedder.random_init_dumps=True",
        "dataset.data_type=idx",
        f"use_cluster={args.cluster}",
        "training.patience=50",
    ])

    conf = runner.conf
    device = conf.training.device
    model_cls = f"{conf.model.classPath}.{conf.model.name}"

    graph_model = get_class(model_cls)(conf, runner.gmn_config).to(device)
    graph_model_best = get_class(model_cls)(conf, runner.gmn_config).to(device)

    input_dim = runner.get_aug_map_input_dim()
    aug_model = AugEmbeddingMap(
        input_dim,
        (conf.aug_embedder.hidden_dim,),
        conf.aug_embedder.output_dim,
        symmetric=conf.aug_embedder.symmetric,
        aug_emb_model=conf.aug_embedder.model,
        model_every_dim=conf.aug_embedder.model_every_dim,
        gen_out_dim=conf.gmn.filters_3,
    ).to(device)

    optimizer = torch.optim.Adam(
        aug_model.parameters(),
        lr=conf.training.learning_rate,
        weight_decay=conf.training.weight_decay,
    )
    runner.model = aug_model
    runner.optimizer = optimizer
    runner.trainer = AugEmbTrainer(
        conf, aug_model, optimizer, runner.gmn_config,
        graph_model, graph_model_best,
        q_embeds_0=runner.q_embeds_0, c_embeds_0=runner.c_embeds_0,
    )
    return runner


def load_aug_checkpoint(cluster_runner: AugEmbRunner) -> None:
    """Load the augnet checkpoint weights into an initialised runner."""
    checkpoint = torch.load(
        cluster_runner.conf.lsh.aug_checkpoint_path,
        map_location=cluster_runner.conf.training.device,
    )
    cluster_runner.model.load_state_dict(checkpoint["model_state_dict"])
    cluster_runner.model.eval()
    cluster_runner.trainer.model = cluster_runner.trainer.model.to(
        cluster_runner.conf.training.device
    )
