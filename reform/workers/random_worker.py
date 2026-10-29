import multiprocessing

import numpy as np
import torch
from loguru import logger

from reform.lsh.constants import LSH_SEED
from reform.utils.retr_sweep_utils import _query_metrics, compute_rerank_scores
from reform.utils.utils import set_seed


def _random_worker(
    q_hat_np: np.ndarray,
    c_hat_np: np.ndarray,
    gt_sets: list[set[int]],
    result_queue: multiprocessing.Queue,
    rerank_mode: str
) -> None:
    SENTINEL = None
    try:
        corpus_size = c_hat_np.shape[0]
        num_queries = q_hat_np.shape[0]

        q_hat = torch.from_numpy(q_hat_np)[:, 0, :]
        c_hat = torch.from_numpy(c_hat_np)[:, 0, :]
        rerank_scores = compute_rerank_scores(q_hat, c_hat, rerank_mode)

        set_seed(LSH_SEED)
        rng = np.random.default_rng(LSH_SEED)

        results = []
        for i in range(1, 101, 10):
            subset_size = int(i / 100 * corpus_size)
            per_query = []
            for q_idx in range(num_queries):
                candidate_ids = rng.choice(corpus_size, size=subset_size, replace=False)

                # Rerank using precomputed cosine similarity scores.
                scores = rerank_scores[q_idx][candidate_ids]  # (|cands|,)
                metrics = _query_metrics(
                    candidate_ids, scores, gt_sets[q_idx], corpus_size
                )
                per_query.append(metrics)

            # Aggregate over queries
            agg = {
                "per_query": per_query,
                **{
                    key: float(np.mean([m[key] for m in per_query]))
                    for key in per_query[0]
                },
            }
            results.append(agg)
            logger.info(
                f"(Random) subset_size={subset_size:2d} "
                f"num_evals={agg['num_evals']:.1f} "
                f"map={agg['map']:.4f} map@50={agg['map@50']:.4f}"
                f"set prec={agg['set_prec']:.4f} set rec={agg['set_rec']:.4f}"
                f"prec@5={agg['prec@5']:.4f} rec@5={agg['rec@5']:.4f}"
                f"mrr={agg['mrr']:.4f}"
            )

        result_queue.put((None, None, results, None, "Random"))

    except Exception as e:
        logger.exception(f"(Random) Worker failed: {e}")
        result_queue.put((None, None, None, None, "Random"))
    finally:
        result_queue.put(SENTINEL)
