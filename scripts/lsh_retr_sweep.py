# ruff: noqa: E402
from dotenv import load_dotenv

# Load environment variables before importing torch
load_dotenv()

import argparse
import os
# Don't mix np multithreading for matmuls with multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import multiprocessing
import operator
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

# ── project root on path ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from reform.lsh.constants import (
    LAMBDA_COLL_MIN_VALUES,
    LAMBDA_FENCE_SIT,
    LINEAR_EVERY_DIM_VALUES,
)
from reform.runner.lsh_run import LSHRunner
from reform.utils.lsh_plots import make_plot_name, plot_sweep_results
from reform.utils.script_utils import (
    CORPUS_SIZE, add_aug_args, _build_ckpt_tag, _build_aug_ckpt_tag,
    noise_dim_suffix as _noise_dim_suffix,
)
from reform.workers.lsh_worker import _lsh_worker
from reform.workers.random_worker import _random_worker


def run_sweep(
    base_overrides: list[str],
    noise_sigma_query: float = 0.0,
    noise_sigma_corpus: float = 0.0,
    noise_seed: int = 0,
    replica_split: bool = False,
) -> None:
    # Instantiate runner to load config, task name, and test embeddings in one shot.
    # wandb_enabled=False avoids initializing a run just for the sweep orchestration.
    dummy_overrides = [
        *base_overrides,
        f"lsh.lambda_fence_sit={LAMBDA_FENCE_SIT}",
        "lsh.lambda_coll_min=0.05",
        "lsh.lambda_bit_bal=0.85",
        "task.wandb_enabled=False",
    ]
    runner = LSHRunner(overrides=dummy_overrides)
    conf = runner.conf

    q_hat = runner.q_hats["val"].cpu()
    c_hat = runner.x_hats["val"].cpu()
    logger.info(f"Val embeddings: q_hat={tuple(q_hat.shape)}, c_hat={tuple(c_hat.shape)}")

    if conf.lsh.subtract_mean:
        mean_val = c_hat.mean(dim=0, keepdim=True)
        q_hat -= mean_val
        c_hat -= mean_val
        logger.info("Subtracted mean from query and corpus embeddings for sweep.")

    append_noise_dim = noise_sigma_query > 0 or noise_sigma_corpus > 0
    if append_noise_dim:
        rng = np.random.default_rng(noise_seed)
        if noise_sigma_query > 0:
            q_noise = torch.tensor(
                rng.normal(0, noise_sigma_query, (*q_hat.shape[:-1], 1)), dtype=q_hat.dtype
            )
            q_hat = torch.cat([q_hat, q_noise], dim=-1)
        if noise_sigma_corpus > 0:
            c_noise = torch.tensor(
                rng.normal(0, noise_sigma_corpus, (*c_hat.shape[:-1], 1)), dtype=c_hat.dtype
            )
            c_hat = torch.cat([c_hat, c_noise], dim=-1)
        logger.info(
            f"Appended noise dim: sigma_query={noise_sigma_query}, sigma_corpus={noise_sigma_corpus}, "
            f"seed={noise_seed} → q_hat={tuple(q_hat.shape)}, c_hat={tuple(c_hat.shape)}"
        )

    # Load val ground truth (conf.dataset.data_type already set to 'idx' by LSHRunner)
    from reform.utils.dataset_loader import SubgraphIsomorphismDataset
    val_dataset = SubgraphIsomorphismDataset(conf, mode="val")
    num_q = q_hat.size(0)
    num_c = c_hat.size(0)
    all_gt = torch.as_tensor(val_dataset.all_gt).reshape(num_q, num_c)
    gt_sets = [
        set(torch.where(all_gt[i] > 0.5)[0].tolist())
        for i in range(num_q)
    ]
    avg_gt = np.mean([len(s) for s in gt_sets])
    logger.info(f"Val GT: {num_q} queries x {num_c} corpus, avg gt size={avg_gt:.1f}")

    # Convert to numpy for pickling across process boundary
    q_hat_np = q_hat.numpy()
    c_hat_np = c_hat.numpy()

    SENTINEL = None
    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue = ctx.Queue()
    procs = []

    if conf.lsh.retr_random_hyper:
        # Lambda and linear_every_dim are model-specific; run a single worker with defaults.
        worker_configs = [(LAMBDA_COLL_MIN_VALUES[0], False)]
    else:
        worker_configs = [
            (lam, linperd)
            for lam in LAMBDA_COLL_MIN_VALUES
            for linperd in LINEAR_EVERY_DIM_VALUES
        ]
    num_workers = len(worker_configs) + 1

    for lambda_coll_min, linear_every_dim in worker_configs:
        p = ctx.Process(
            target=_lsh_worker,
            args=(
                base_overrides, lambda_coll_min, linear_every_dim,
                q_hat_np, c_hat_np, gt_sets, result_queue, append_noise_dim,
                None, replica_split,
            ),
        )
        procs.append(p)
        p.start()

    rp = ctx.Process(
        target=_random_worker,
        args=(q_hat_np, c_hat_np, gt_sets, result_queue, conf.lsh.rerank_mode),
    )
    procs.append(rp)
    rp.start()

    all_results: dict[tuple[float | None, bool | None, str], list[dict] | None] = {}
    all_hyperplanes: dict[tuple[float | None, bool | None, str], dict[int, dict] | None] = {}
    seen_sentinels = 0
    while seen_sentinels < num_workers:
        item = result_queue.get()
        if item is SENTINEL:
            seen_sentinels += 1
        else:
            lam, linperd, res, hyperplanes, worker_type = item
            all_results[lam, linperd, worker_type] = res
            all_hyperplanes[lam, linperd, worker_type] = hyperplanes

    for p in procs:
        p.join()

    sub_mean_suffix = "_sub_mean" if conf.lsh.subtract_mean else ""
    if conf.lsh.retr_random_hyper:
        hyper_suffix = f"_random_hyper_seed{conf.lsh.random_hyper_seed}"
    elif conf.lsh.third_loss_type != 'sc_coll_min':
        hyper_suffix = f"_{conf.lsh.third_loss_type}"
    else:
        hyper_suffix = ""
    nd_sfx    = _noise_dim_suffix(noise_sigma_query, noise_sigma_corpus)
    plot_name = make_plot_name(conf.dataset.name, conf.task.name, rerank_mode=conf.lsh.rerank_mode)
    dump_dir  = Path("./allPklDumps") / conf.dataset.name
    out_fp = dump_dir / f"lsh_retr_sweep_{plot_name}{sub_mean_suffix}{hyper_suffix}{nd_sfx}.pkl"
    with open(out_fp, "wb") as f:
        pickle.dump(all_results, f)
    logger.info(f"Saved results → {out_fp}")

    hp_fp = dump_dir / f"lsh_hyperplanes_{plot_name}{sub_mean_suffix}{hyper_suffix}{nd_sfx}.pkl"
    with open(hp_fp, "wb") as f:
        pickle.dump(all_hyperplanes, f)
    logger.info(f"Saved hyperplanes → {hp_fp}")

    trained_loss_type = None if conf.lsh.retr_random_hyper else conf.lsh.third_loss_type
    plot_dir = plot_sweep_results(
        all_results, conf.base_dir, conf.dataset.name, conf.task.name, base_overrides,
        inner_dir=conf.lsh.inner_plot_dir,
        random_hyper_seed=conf.lsh.random_hyper_seed if conf.lsh.retr_random_hyper else None,
        trained_loss_type=trained_loss_type,
        noise_sigma_query=noise_sigma_query,
        noise_sigma_corpus=noise_sigma_corpus,
        rerank_mode=conf.lsh.rerank_mode,
    )
    logger.info(f"Saved plots → {plot_dir}")

    for lam, linperd in worker_configs:
        res = all_results.get((lam, linperd, "LSH"))
        if res is None:
            logger.warning(f"λ={lam} linperd={linperd}: FAILED")
            continue
        best = max(res, key=operator.itemgetter("map"))
        logger.info(
            f"λ={lam:.2f} linperd={linperd}: best map={best['map']:.4f}"
            f" at subset_size={best['subset_size']} (num_evals={best['num_evals']:.1f})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="LSH random hyperplane retrieval sweep over subset sizes and lambda values."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g. ptc_fr)")
    parser.add_argument("--gpu", type=str, default="3",
                        help="CUDA_VISIBLE_DEVICES value (default: 3)")
    parser.add_argument("--random-hyper-seed", type=int, default=0,
                        help="Seed for random hyperplane generation (default: 0)")
    parser.add_argument("--subtract-mean", action="store_true", default=False,
                        help="Subtract corpus mean from query and corpus embeddings before sweep (default: False)")
    parser.add_argument("--rerank-mode", type=str, default="cos", choices=["cos", "orig"],
                        help="Reranking mode for the random baseline worker (default: cos)")
    parser.add_argument("--trained-hyper", action="store_true", default=False,
                        help="Use trained model weights instead of random Gaussian hyperplanes (default: False)")
    parser.add_argument("--inner-plot-dir", type=str, default=None,
                        help="Subdirectory inside the plot output dir. "
                             "Defaults to 'rh_lsh' for random hyperplanes, 'trained_lsh' for trained.")
    parser.add_argument("--third-loss-type", type=str, default="sc_coll_min",
                        choices=["sc_coll_min", "ad_decorr"],
                        help="Third loss type used during LSH training (default: sc_coll_min)")
    parser.add_argument("--noise-sigma-query", type=float, default=0.0,
                        help="Std of noise dimension appended to query embeddings (default: 0 = disabled)")
    parser.add_argument("--noise-sigma-corpus", type=float, default=0.0,
                        help="Std of noise dimension appended to corpus embeddings (default: 0 = disabled)")
    parser.add_argument("--noise-seed", type=int, default=0,
                        help="RNG seed for noise dimension sampling (default: 0)")
    parser.add_argument("--pre-augnet", action="store_true", default=False,
                        help="Load pre-AugNet (random-feature-map) embeddings instead of AugNet MLP outputs")
    parser.add_argument("--replica-split", action="store_true", default=False,
                        help="Implies --pre-augnet; reshapes (D,N,K) → (D*R,N,K/R) so each hyperplane "
                             "gets its own hash table (DISCo multi-replica union). "
                             "Output files use _pre_augnet_replicas suffix instead of _pre_augnet.")
    parser.add_argument("--random-init", action="store_true", default=False,
                        help="Load embeddings from randomly-initialized GNN+AugNet dumps "
                             "(_dump_random_init_embs). No checkpoint loaded, no replica reshape.")
    parser.add_argument(
        "--legacy-ptc-fr",
        action="store_true",
        default=False,
        help=(
            "Reproduce the original ptc_fr random-hyperplane sweep results from Apr 2026. "
            "\n\n"
            "Background: before commit bdef568 (Apr 19 2026), lsh_retr_sweep did not pass "
            "early_stopping.tolerance as a Hydra override, so the Hydra task name never "
            "acquired the '_estol=1e-05' suffix. This caused LSHRunner to look for dump dirs "
            "without that suffix. For ptc_fr specifically, those non-estol dump dirs were "
            "created on Apr 7 from latestModels/.../0.pt (epoch-0, untrained AugNet weights) "
            "rather than bestValidationModels/best.pt — a separate bug in the aug_map_dump "
            "invocation at the time. All other datasets' non-estol dump dirs correctly used "
            "bestValidationModels/best.pt and are unaffected. "
            "\n\n"
            "Effect: the random-LSH sweep evaluated on epoch-0 embeddings, selecting a "
            "potentially suboptimal k/C ratio. The downstream data generation was NOT affected "
            "(gen_dataset_carefully.py loads the AugNet checkpoint directly and computes "
            "embeddings fresh). The trained-LSH pipeline WAS affected because lsh_sweep also "
            "loaded from the same dump dirs to train hyperplanes. "
            "\n\n"
            "When this flag is set, early_stopping.tolerance is NOT added to base_overrides, "
            "so LSHRunner resolves the same non-estol dump dirs as the original Apr 2026 runs, "
            "reproducing identical sweep results. Do NOT use this for new experiments — it "
            "intentionally loads from the incorrect (epoch-0) dump dirs."
        ),
    )
    add_aug_args(parser)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.inner_plot_dir is None:
        args.inner_plot_dir = "trained_lsh" if args.trained_hyper else "rh_lsh"

    ckpt_tag     = _build_ckpt_tag(args)
    aug_ckpt_tag = _build_aug_ckpt_tag(args)

    if args.random_init:
        augnet_overrides = [
            "aug_embedder.use_pretrained_embs=False",
            "aug_embedder.random_init_dumps=True",
            "aug_embedder.pre_augnet_dumps=False",
            "aug_embedder.pre_augnet_replicas=False",
        ]
    else:
        augnet_overrides = [
            f"aug_embedder.best_checkpoint_path=bestValidationModels/{ckpt_tag}/best.pt",
            f"aug_embedder.latest_ckpts_path=latestModels/{ckpt_tag}",
            f"lsh.aug_checkpoint_path=bestValidationModels/{aug_ckpt_tag}/best.pt",
            "aug_embedder.use_pretrained_embs=True",
            f"aug_embedder.pre_augnet_dumps={args.pre_augnet or args.replica_split}",
            f"aug_embedder.pre_augnet_replicas={args.replica_split}",
            *([] if args.legacy_ptc_fr else [f"early_stopping.tolerance={args.estol}"]),
        ]

    base_overrides = [
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
        f"lsh.random_hyper_seed={args.random_hyper_seed}",
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
        f"lsh.subtract_mean={args.subtract_mean}",
        f"lsh.rerank_mode={args.rerank_mode}",
        f"lsh.retr_random_hyper={not args.trained_hyper}",
        f"lsh.third_loss_type={args.third_loss_type}",
        f"lsh.inner_plot_dir={args.inner_plot_dir}",
        "dataset.data_type=idx",
        "use_cluster=0",
        "task.wandb_enabled=false",
        *augnet_overrides,
    ]

    run_sweep(
        base_overrides,
        noise_sigma_query=args.noise_sigma_query,
        noise_sigma_corpus=args.noise_sigma_corpus,
        noise_seed=args.noise_seed,
        replica_split=args.replica_split and not args.random_init,
    )


if __name__ == "__main__":
    main()
