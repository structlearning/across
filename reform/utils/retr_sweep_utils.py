# add all necessary imports
import multiprocessing

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from reform.lsh.constants import TOP_KS
from reform.utils.training_utils import (
    custom_ap,
    custom_ap_at_K,
    precision_at_k,
    recall_at_k,
    compute_mrr
)


def _query_metrics(
    retrieved_ids: list[int],
    retrieved_scores: np.ndarray,
    gt_set: set[int],
    corpus_size: int,
) -> dict:
    num_evals = len(retrieved_ids)

    if num_evals == 0:
        metrics = {"num_evals": 0, "map": 0.0, "map@50": 0.0,
                   "set_prec": 0.0, "set_rec": 0.0, "mrr": 0.0}
        for k in TOP_KS:
            metrics[f"prec@{k}"] = 0.0
            metrics[f"rec@{k}"] = 0.0
        return metrics

    # Full-corpus score array: -inf for non-retrieved items
    full_scores = np.full(corpus_size, -np.inf, dtype=np.float32)
    for cid, sc in zip(retrieved_ids, retrieved_scores, strict=True):
        full_scores[cid] = float(sc)

    # Binary relevance vector over full corpus
    y_true = np.zeros(corpus_size, dtype=np.float32)
    for cid in gt_set:
        y_true[cid] = 1.0

    ap = custom_ap(gt_set, retrieved_ids, list(retrieved_scores), K=-1, len_gt=len(gt_set))
    ap50 = custom_ap_at_K(gt_set, retrieved_ids, list(retrieved_scores), K=50, len_gt=len(gt_set))

    metrics = {"num_evals": num_evals, "map": ap, "map@50": ap50}
    metrics["set_prec"] = precision_at_k(y_true, full_scores)
    metrics["set_rec"] = recall_at_k(y_true, full_scores)
    metrics["mrr"] = compute_mrr(y_true, full_scores)

    # y_true_long_np = y_true.astype(np.int64)
    for k in TOP_KS:
        # metrics[f"ndcg@{k}"] = compute_ndcg(y_true_long_np, retrieved_ids, retrieved_scores, K=k)
        metrics[f"prec@{k}"] = precision_at_k(y_true, full_scores, k=k)
        metrics[f"rec@{k}"] = recall_at_k(y_true, full_scores, k=k)

    return metrics


def compute_rerank_scores(q_pre: torch.Tensor, c_pre: torch.Tensor, rerank_mode: str) -> np.ndarray:
    if rerank_mode == "cos":
        q_pre_norm = F.normalize(q_pre.float(), dim=-1)  # (num_q, d_pre)
        c_pre_norm = F.normalize(c_pre.float(), dim=-1)  # (N, d_pre)
        rerank_scores = (q_pre_norm @ c_pre_norm.T).numpy()  # (num_q, N)

    elif rerank_mode == "orig":
        # For any query q_i and corpus c_i pair, the original score
        # looks like -|| |q_i - c_i|_+ ||_1, where |.|_+ is ReLU.
        rerank_scores = -torch.norm(F.relu(q_pre[:, None] - c_pre[None]), p=1, dim=-1).numpy()  # (num_q, N)
    else:
        raise ValueError(f"Unknown rerank_mode: {rerank_mode}")

    return rerank_scores