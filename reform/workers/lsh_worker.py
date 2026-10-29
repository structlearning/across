# add all necessary imports
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from reform.lsh import LSH, create_bit_indices
from reform.lsh.constants import LAMBDA_FENCE_SIT, LSH_SEED, SUBSET_SIZES
from reform.models import HashModel
from reform.runner.fmap_run import FMapRunner
from reform.runner.lsh_run import LSHRunner, _build_conf_only
from reform.utils.retr_sweep_utils import _query_metrics, compute_rerank_scores
from reform.utils.utils import set_seed


def _lsh_worker(
    base_overrides: list[str],
    lambda_coll_min: float,
    linear_every_dim: bool,
    q_hat_np: np.ndarray,
    c_hat_np: np.ndarray,
    gt_sets: list[set[int]],
    result_queue: multiprocessing.Queue,
    append_noise_dim: bool = False,
    runner_cls=None,
    replica_split: bool = False,
) -> None:
    SENTINEL = None
    try:
        lambda_bit_bal = round(1.0 - LAMBDA_FENCE_SIT - lambda_coll_min, 10)
        overrides = [
            *base_overrides,
            f"lsh.lambda_fence_sit={LAMBDA_FENCE_SIT}",
            f"lsh.lambda_coll_min={lambda_coll_min}",
            f"lsh.lambda_bit_bal={lambda_bit_bal}",
            f"lsh.linear_every_dim={linear_every_dim}",
            "task.wandb_enabled=False",
        ]
        _runner_cls = runner_cls or LSHRunner
        conf = _build_conf_only(_runner_cls, overrides)
        _is_fmap = issubclass(_runner_cls, FMapRunner)
        embd_dim = conf.fmap_embedder.output_dim if _is_fmap else q_hat_np.shape[-1]

        model = None
        if not conf.lsh.retr_random_hyper:
            ckpt = Path(conf.base_dir) / conf.early_stopping.best_val_name / conf.task.name / "best.pt"
            logger.info(f"Loading from checkpoint: λ_coll_min={lambda_coll_min} linear_every_dim={linear_every_dim} ckpt={ckpt}")

            if not ckpt.exists():
                logger.warning(
                    f"[λ={lambda_coll_min} linperd={linear_every_dim}] (LSH) Checkpoint not found: {ckpt}"
                )
                result_queue.put((lambda_coll_min, linear_every_dim, None, None, "LSH"))
                result_queue.put(SENTINEL)
                return

            model = HashModel(embd_dim, conf.lsh.output_dim,
                              conf.gmn.filters_3, linear_every_dim=conf.lsh.linear_every_dim)
            state = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(state["model_state_dict"])
            model.eval()
            logger.info(
                f"[λ={lambda_coll_min} linperd={linear_every_dim}] (LSH) Loaded checkpoint from epoch {state.get('epoch', '?')}"
            )
        else:
            logger.info(
                f"[λ={lambda_coll_min} linperd={linear_every_dim}] (LSH) Using random hyperplanes"
            )

        q_hat = torch.from_numpy(q_hat_np)
        c_hat = torch.from_numpy(c_hat_np)
        corpus_size = c_hat.size(0)
        num_queries = q_hat.size(0)

        gen_embd_dim = conf.gmn.filters_3

        # Rerank scores use the full K-dim vector; compute before any replica reshape.
        rerank_scores = compute_rerank_scores(
            q_hat[:, 0, :], c_hat[:, 0, :], conf.lsh.rerank_mode
        )

        # c_hat: (N, gen_embd_dim, d) → (gen_embd_dim, N, d)
        # Permutation done as LSH code expects this.
        c_lsh = c_hat.permute(1, 0, 2)

        if replica_split:
            # Each of the R hyperplane contributions is a contiguous K/R chunk.
            # Reshape to give each its own hash table: (N,D,K) → (N,D*R,K/R) → (D*R,N,K/R).
            R = conf.aug_embedder.num_hyperplanes
            D_orig = conf.gmn.filters_3
            K_full = q_hat.size(-1)
            embd_dim = K_full // R
            gen_embd_dim = D_orig * R
            c_lsh = c_hat.view(corpus_size, D_orig, R, embd_dim).view(corpus_size, gen_embd_dim, embd_dim).permute(1, 0, 2)
            q_hat = q_hat.view(num_queries, D_orig, R, embd_dim).view(num_queries, gen_embd_dim, embd_dim)

        subset_sizes = list(range(conf.lsh.output_dim)) if conf.lsh.retr_random_hyper else SUBSET_SIZES
        subset_results = []
        # RH: {subset_size: {hp_weights, hp_biases, bit_indices}}
        # Trained: {hp_weights, hp_biases, bit_indices_by_subset: {subset_size: bit_indices}}
        hyperplanes_by_subset: dict = {}
        trained_bit_indices: dict[int, Any] = {}
        effective_seed = LSH_SEED + conf.lsh.random_hyper_seed
        for subset_size in subset_sizes:
            set_seed(effective_seed)
            rng = np.random.default_rng(effective_seed)

            bit_indices = create_bit_indices(
                hashcode_dim=conf.lsh.output_dim,
                subset_size=subset_size,
                gen_embd_dim=gen_embd_dim,
                rng=rng,
                same_idxs_across_dims=conf.lsh.same_idxs_across_dims,
            )
            if conf.lsh.retr_random_hyper:
                lsh = LSH.from_random(
                    hashcode_dim=conf.lsh.output_dim,
                    embd_dim=embd_dim + 1 if append_noise_dim else embd_dim,
                    bit_indices=bit_indices,
                    corpus=c_lsh,
                    rng=rng,
                    linear_every_dim=conf.lsh.linear_every_dim,
                )
                hyperplanes_by_subset[subset_size] = {
                    "hp_weights": lsh.hp_weights.numpy(),
                    "hp_biases": lsh.hp_biases.numpy(),
                    "bit_indices": bit_indices,
                }
            else:
                lsh = LSH.from_model(model.linears, bit_indices, c_lsh)
                if not hyperplanes_by_subset:   # capture weights once
                    hyperplanes_by_subset["hp_weights"] = lsh.hp_weights.numpy()
                    hyperplanes_by_subset["hp_biases"]  = lsh.hp_biases.numpy()
                trained_bit_indices[subset_size] = bit_indices

            per_query = []
            for q_idx in range(num_queries):
                q_lsh = q_hat[q_idx]  # (gen_embd_dim, d)

                candidate_ids = sorted(lsh.retrieve(q_lsh))

                # Rerank using precomputed cosine similarity scores.
                scores = rerank_scores[q_idx][candidate_ids]  # (|cands|,)

                metrics = _query_metrics(
                    candidate_ids, scores, gt_sets[q_idx], corpus_size
                )
                per_query.append(metrics)

            # Aggregate over queries
            agg = {
                "subset_size": subset_size,
                "per_query": per_query,
                **{
                    key: float(np.mean([m[key] for m in per_query]))
                    for key in per_query[0]
                    if key != "subset_size"
                },
            }
            subset_results.append(agg)
            logger.info(
                f"[λ={lambda_coll_min} linperd={linear_every_dim}] (LSH) subset_size={subset_size:2d} "
                f"num_evals={agg['num_evals']:.1f} "
                f"map={agg['map']:.4f} map@50={agg['map@50']:.4f}"
                f"set prec={agg['set_prec']:.4f} set rec={agg['set_rec']:.4f}"
                f"prec@5={agg['prec@5']:.4f} rec@5={agg['rec@5']:.4f}"
                f"mrr={agg['mrr']:.4f}"
            )

        if not conf.lsh.retr_random_hyper:
            hyperplanes_by_subset["bit_indices_by_subset"] = trained_bit_indices

        result_queue.put((lambda_coll_min, linear_every_dim, subset_results, hyperplanes_by_subset, "LSH"))

    except Exception as e:
        logger.exception(f"[λ={lambda_coll_min} linperd={linear_every_dim}] (LSH) Worker failed: {e}")
        result_queue.put((lambda_coll_min, linear_every_dim, None, None, "LSH"))
    finally:
        result_queue.put(SENTINEL)
