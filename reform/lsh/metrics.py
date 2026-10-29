from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor

from .constants import TOP_KS

ModelConfig = tuple[bool, float]
IdxConfig = tuple[int, bool]


@dataclass(slots=True, frozen=True)
class Metrics:
    fraction_retrieved: float
    recall: float
    precision: float
    mrr: float
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    map: float
    map_at_50: float


@dataclass(slots=True, frozen=True)
class Result:
    model_config: ModelConfig
    idx_config: IdxConfig
    metrics: Metrics


def compute_metrics(retrieved: list[Tensor], cosine_sims: Tensor, gt: Tensor) -> Metrics:
    running_num_retrieved = 0.0
    running_precision_at_k = defaultdict(lambda: 0.0)
    running_recall_at_k = defaultdict(lambda: 0.0)
    running_precision = 0.0
    running_recall = 0.0
    running_map = 0.0
    running_map_at_50 = 0.0
    running_mrr = 0.0

    relevant_counts = torch.sum(gt, dim=-1)
    for i, (c_ids, all_scores) in enumerate(zip(retrieved, cosine_sims, strict=True)):
        scores = all_scores[c_ids]
        idxs = torch.argsort(scores, descending=True)
        hits = gt[i, c_ids[idxs]]
        num_retrieved = len(c_ids)
        running_num_retrieved += num_retrieved
        num_relevant = relevant_counts[i].item()
        if num_relevant == 0:
            continue
        num_relevant_in_k = torch.cumsum(hits, dim=0)
        for k in TOP_KS:
            idx = min(k, num_retrieved) - 1
            rel_at_k = num_relevant_in_k[idx].item()
            running_precision_at_k[k] += rel_at_k / k
            running_recall_at_k[k] += rel_at_k / num_relevant
        running_precision += num_relevant_in_k[-1].item() / num_retrieved
        running_recall += num_relevant_in_k[-1].item() / num_relevant
        running_map += (
            torch.sum(
                hits * num_relevant_in_k / torch.arange(1, num_retrieved + 1, dtype=torch.float32)
            ).item()
            / num_relevant
        )
        x = min(50, num_retrieved)
        running_map_at_50 += (
            torch.sum(
                hits[:50] * num_relevant_in_k[:50] / torch.arange(1, x + 1, dtype=torch.float32)
            )
            / num_relevant_in_k[x - 1]
        ).item()
        # don't see a way to short-circuit and return the index of the first 1 in PyTorch/Numpy
        # .any() is a fast enough way to prevent Python from for-looping over
        # the whole thing when there is no hit
        # hits are generally close to the beginning so a Python for loop is the
        # fastest (unless some jit compilation is done)
        if hits.any():
            for r in range(len(hits)):
                if hits[r]:
                    running_mrr += 1.0 / (1 + r)
                    break

    total = len(retrieved)
    return Metrics(
        fraction_retrieved=running_num_retrieved / gt.size(1) / total,
        recall=running_recall / total,
        precision=running_precision / total,
        recall_at_k={k: v / total for k, v in running_recall_at_k.items()},
        precision_at_k={k: v / total for k, v in running_precision_at_k.items()},
        map=running_map / total,
        map_at_50=running_map_at_50 / total,
        mrr=running_mrr / total,
    )
