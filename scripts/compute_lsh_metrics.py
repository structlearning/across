# ruff: noqa: E402
from dotenv import load_dotenv

# Load environment variables before importing torch
load_dotenv()

import os

# We are using numpy/torch in multiple child processes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pickle
import sys
import typing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import astuple, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor

from reform.lsh import LSH, create_bit_indices
from reform.lsh.constants import (
    LAMBDA_COLL_MIN_VALUES,
    LAMBDA_FENCE_SIT,
    LINEAR_EVERY_DIM_VALUES,
    SAME_IDXS_ACROSS_DIMS_VALUES,
    SUBSET_SIZES,
)
from reform.lsh.metrics import IdxConfig, Metrics, ModelConfig, Result, compute_metrics
from reform.runner.lsh_run import LSHRunner
from reform.runner.lsh_run import _build_conf_only as build_foo

NUM_JOBS = 32

logger.disable("reform")


@dataclass(slots=True, frozen=True)
class LSHTask:
    linear_every_dim: bool
    lambda_coll: float
    subset_size: int
    same_idxs_across_dims: bool


TASKS = [
    LSHTask(linear_every_dim, lambda_coll, subset_size, same_idxs_across_dims)
    for linear_every_dim in LINEAR_EVERY_DIM_VALUES
    for lambda_coll in LAMBDA_COLL_MIN_VALUES
    for subset_size in SUBSET_SIZES
    for same_idxs_across_dims in SAME_IDXS_ACROSS_DIMS_VALUES
]


def override_foo(
    base_overrides: list[str], linear_every_dim: bool, lambda_coll: float
) -> list[str]:
    lambda_bit_bal = round(1.0 - LAMBDA_FENCE_SIT - lambda_coll, 10)
    return [
        *base_overrides,
        f"lsh.lambda_fence_sit={LAMBDA_FENCE_SIT}",
        f"lsh.lambda_coll_min={lambda_coll}",
        f"lsh.lambda_bit_bal={lambda_bit_bal}",
        f"lsh.linear_every_dim={linear_every_dim}",
        "task.wandb_enabled=False",
    ]


class CheckpointPathCtor:
    def __init__(self, base_overrides: list[str]):
        self.name_template = "{base_dir}/{best_val_name}/{task_name}/best.pt"
        self.base_overrides = base_overrides

    def __call__(self, linear_every_dim: bool, lambda_coll: float) -> str:
        overrides = override_foo(self.base_overrides, linear_every_dim, lambda_coll)
        conf = build_foo(LSHRunner, overrides)
        return self.name_template.format(
            base_dir=conf.base_dir,
            best_val_name=conf.early_stopping.best_val_name,
            task_name=conf.task.name,
        )


def setup_base_dir(base_dir: Path) -> Path:
    timestamp = (
        datetime
        .now(timezone.utc)
        .astimezone()
        .isoformat(timespec="microseconds")
        .replace(":", "-")
    )
    (base_dir / timestamp).mkdir(parents=True, exist_ok=True)
    latest = base_dir / "latest"
    if latest.is_symlink():
        latest.unlink()
    latest.symlink_to(timestamp)
    return base_dir / timestamp


def load_query_and_corpus_aug_embds(base_overrides: list[str]) -> tuple[Tensor, Tensor]:
    runner = LSHRunner(overrides=override_foo(base_overrides, False, 0))  # dummy args

    def permute(x: Tensor) -> Tensor:
        return x.permute(1, 0, 2).contiguous()

    return permute(runner.q_hats["val"].cpu()), permute(runner.x_hats["val"].cpu())


def load_linear_layers(checkpoint) -> nn.Linear | nn.ModuleList:
    linears_state = {
        k[len("linears.") :]: v for k, v in checkpoint.items() if k.startswith("linears.")
    }
    is_module_list = any(k.startswith("0.") for k in linears_state)
    if is_module_list:
        n = max(int(k.split(".")[0]) for k in linears_state) + 1
        out_dim, in_dim = linears_state["0.weight"].shape
        model = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(n)])
    else:
        out_dim, in_dim = linears_state["weight"].shape
        model = nn.Linear(in_dim, out_dim)
    model.load_state_dict(linears_state)
    return model


MetricsCollection = dict[ModelConfig, dict[IdxConfig, Metrics]]

c_aug_embds: Tensor
q_aug_embds: Tensor
c_gen_embds: Tensor
q_gen_embds: Tensor
cosine_sims: Tensor
gt: Tensor
ckpt_path_ctor: CheckpointPathCtor

model_cache: dict[ModelConfig, nn.Linear | nn.ModuleList] = {}
"""
To prevent the same worker from loading the same model twice.

Non-ancient Python does not need multiprocessing to do basic stuff. In ancient
Python 3.10, there is no alternative to multiprocessing even for simple stuff.
No point optimizing this any more than this: Multiple processes will load the
same model again and again; the fix is to not use ancient Python and use free
threading Python.
"""


def init_worker(
    c_aug_embds_: Tensor,
    q_aug_embds_: Tensor,
    cosine_sims_: Tensor,
    gt_: Tensor,
    ckpt_path_ctor_: CheckpointPathCtor,
):
    global c_aug_embds, q_aug_embds, cosine_sims, gt, ckpt_path_ctor
    c_aug_embds = c_aug_embds_.clone()
    q_aug_embds = q_aug_embds_.clone()
    gt = gt_.bool().clone()
    cosine_sims = cosine_sims_.clone()
    ckpt_path_ctor = ckpt_path_ctor_


def worker(
    linear_every_dim: bool,
    lambda_coll: float,
    subset_size: int,
    same_idxs_across_dims: bool,
) -> Metrics:
    seed = hash((linear_every_dim, lambda_coll, subset_size, same_idxs_across_dims)) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)

    if not (model := model_cache.get((linear_every_dim, lambda_coll))):
        checkpoint = torch.load(ckpt_path_ctor(linear_every_dim, lambda_coll))["model_state_dict"]
        model = load_linear_layers(checkpoint)

    hashcode_dim = typing.cast(
        int, model[0].out_features if isinstance(model, nn.ModuleList) else model.out_features
    )
    bit_indices = create_bit_indices(
        hashcode_dim=hashcode_dim,
        subset_size=subset_size,
        gen_embd_dim=c_aug_embds.size(0),
        rng=rng,
        same_idxs_across_dims=same_idxs_across_dims,
    )
    lsh = LSH.from_model(model, bit_indices, c_aug_embds)

    return compute_metrics(
        [
            torch.tensor(sorted(lsh.retrieve(q_aug_embds[:, i, :])))
            for i in range(q_aug_embds.size(1))
        ],
        cosine_sims,
        gt,
    )


def compute_metrics_for_all_configs(
    queries: Tensor,
    corpus: Tensor,
    cosine_sims: Tensor,
    gt: Tensor,
    base_overrides: list[str],
) -> list[Result]:
    results: list[Result] = []
    with ProcessPoolExecutor(
        max_workers=NUM_JOBS,
        initializer=init_worker,
        initargs=(corpus, queries, cosine_sims, gt, CheckpointPathCtor(base_overrides)),
    ) as executor:
        future_to_task = {executor.submit(worker, *astuple(task)): task for task in TASKS}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            logger.info(
                f"Completed {task.linear_every_dim=}, {task.lambda_coll=}, "
                f"{task.subset_size=}, {task.same_idxs_across_dims=}"
            )
            results.append(
                Result(
                    model_config=(task.linear_every_dim, task.lambda_coll),
                    idx_config=(task.subset_size, task.same_idxs_across_dims),
                    metrics=future.result(),
                )
            )

    return results


def main():
    mp.set_start_method("spawn")

    base_overrides = sys.argv[1:]
    base_dir = setup_base_dir(Path("artifacts/lsh"))

    q_aug_embds, c_aug_embds = load_query_and_corpus_aug_embds(base_overrides)
    obj = torch.load("artifacts/gen_embeddings.pt")
    gt = torch.load("artifacts/gt.pt")
    q_gen_embds = F.normalize(obj["q_embeds"], dim=-1)
    c_gen_embds = F.normalize(obj["c_embeds"], dim=-1)
    cosine_sims = q_gen_embds @ c_gen_embds.T
    logger.info(f"Loaded corpus embeddings with size {c_aug_embds.size()}")

    metrics = compute_metrics_for_all_configs(
        q_aug_embds, c_aug_embds, cosine_sims, gt, base_overrides
    )
    logger.info("Creation and retrieval complete")

    with open(base_dir / "data.pkl", "wb") as f:
        pickle.dump(metrics, f)
    logger.info("Saved retrieved items")


if __name__ == "__main__":
    main()
