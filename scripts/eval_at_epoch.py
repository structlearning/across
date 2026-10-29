"""Evaluate the best GNN checkpoint at or before a given epoch cutoff.

For a given gen_run command, finds the checkpoint at or before `epoch_pt` with
the highest validation metric, then runs test evaluation and writes results to
testMAPs/<task>.txt.

Usage
-----
python -B scripts/eval_at_epoch.py \\
    --epoch_pt 200 \\
    [--metric ap_score] \\
    dataset.name=ptc_fr model.name=GEN dataset.rel_mode=sub_iso \\
    training.batch_size=128 use_cluster=1 training.relationship_load=infomax \\
    cluster_suffix=all_corpus_negs_K=40_ratio=0.13_alpha=0.3 \\
    training.loss=celoss training.stratified_batching=False \\
    model.scoring_function=hinge \\
    early_stopping.patience=200 early_stopping.metric=ap_score
"""

import argparse
import hashlib
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from loguru import logger

from reform.runner.gen_run import GENRunner
from reform.training.early_stopping import EarlyStoppingModule


def parse_log_val_metrics(log_path: str, metric: str) -> dict[int, float]:
    """Return {epoch: val_score} from the training log. Last entry wins per epoch."""
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Training log not found: {log_path}")
    scores = {}
    pat = re.compile(r"Run: (\d+) VAL ap_score: ([\d.e+\-]+) map_score: ([\d.e+\-]+)")
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                epoch = int(m.group(1))
                val = float(m.group(2)) if metric == "ap_score" else float(m.group(3))
                scores[epoch] = val
    return scores


def find_best_checkpoint(latest_dir: str, epoch_pt: int, val_scores: dict[int, float],
                         delta: float) -> tuple[int, float]:
    """Replay EarlyStoppingModule logic up to epoch_pt and return (best_epoch, best_score).

    Uses a temp dir so ES can do its normal file saves without touching real checkpoints.
    An epoch becomes the new best when score - prev_best >= delta, exactly as ES does.
    """
    available = {
        int(f[:-3]) for f in os.listdir(latest_dir)
        if f.endswith(".pt") and f[:-3].isdigit()
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        es = EarlyStoppingModule(
            save_dir=tmpdir,
            run_id="eval",
            patience=999999,
            delta=delta,
            logger=logger,
            best_val_name="bestValidationModels",
        )

        best_epoch = None
        best_score = None

        for epoch in sorted(val_scores):
            if epoch > epoch_pt:
                break
            score = val_scores[epoch]
            prev_best = list(es.best_scores) if es.best_scores is not None else None
            es.check([score], {"epoch": epoch})
            if es.best_scores != prev_best:
                best_epoch = epoch
                best_score = score

    if best_epoch is None:
        raise RuntimeError(f"No VAL entries at or before epoch {epoch_pt} in log.")
    if best_epoch not in available:
        raise RuntimeError(
            f"Best epoch {best_epoch} (val={best_score:.6f}) has no checkpoint in {latest_dir}.\n"
            f"Available: {sorted(available)[:20]}"
        )
    return best_epoch, best_score


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--epoch_pt", type=int, required=True,
                        help="Epoch cutoff — evaluate the best checkpoint at or before this epoch.")
    parser.add_argument("--metric", type=str, default=None,
                        help="Validation metric: ap_score or map_score. "
                             "Defaults to early_stopping.metric from config.")
    parser.add_argument("--estol", type=float, default=None,
                        help="Early stopping tolerance (delta). "
                             "Defaults to early_stopping.tolerance from config.")
    script_args, overrides = parser.parse_known_args()

    # Disable wandb and resume — this script only evaluates, never trains
    if not any(o.startswith("task.wandb_enabled") for o in overrides):
        overrides.append("task.wandb_enabled=false")
    if not any(o.startswith("training.resume") for o in overrides):
        overrides.append("training.resume=false")

    runner = GENRunner(overrides=overrides)
    runner.set_trainer()
    conf   = runner.conf
    trainer = runner.trainer

    metric = script_args.metric or conf.early_stopping.metric
    delta  = script_args.estol  if script_args.estol is not None else conf.early_stopping.tolerance
    task   = conf.task.name

    log_path   = os.path.join(conf.log.dir, task + ".log")
    latest_dir = os.path.join(conf.base_dir, "latestModels", task)

    # Fallback: some runs trained with a transient version that truncated the
    # whole task name string (not per-component) when total bytes exceeded 255.
    if not os.path.exists(log_path) or not os.path.isdir(latest_dir):
        if len(task.encode()) > 255:
            name_hash = hashlib.md5(task.encode()).hexdigest()[:8]
            legacy_task = task[:240] + "_" + name_hash
            legacy_log   = os.path.join(conf.log.dir, legacy_task + ".log")
            legacy_ldir  = os.path.join(conf.base_dir, "latestModels", legacy_task)
            if os.path.exists(legacy_log) or os.path.isdir(legacy_ldir):
                logger.warning(
                    f"Primary paths not found; falling back to legacy whole-string truncation.\n"
                    f"  log:     {legacy_log}\n"
                    f"  latest:  {legacy_ldir}"
                )
                task       = legacy_task
                log_path   = legacy_log
                latest_dir = legacy_ldir

    if not os.path.isdir(latest_dir):
        raise RuntimeError(f"latestModels dir not found: {latest_dir}")

    logger.info(f"Task:        {task}")
    logger.info(f"Log:         {log_path}")
    logger.info(f"Checkpoints: {latest_dir}")
    logger.info(f"epoch_pt={script_args.epoch_pt}  metric={metric}  delta={delta}")

    val_scores = parse_log_val_metrics(log_path, metric)
    if not val_scores:
        raise RuntimeError(f"Log exists but contains no VAL lines: {log_path}")
    logger.info(f"Parsed {len(val_scores)} VAL entries (epochs {min(val_scores)}–{max(val_scores)})")

    best_epoch, best_val = find_best_checkpoint(latest_dir, script_args.epoch_pt, val_scores, delta)
    logger.info(f"Best checkpoint: epoch={best_epoch}  val_{metric}={best_val:.6f}")

    ckpt_path = os.path.join(latest_dir, f"{best_epoch}.pt")
    ckpt = torch.load(ckpt_path, map_location=conf.training.device)
    trainer.model.load_state_dict(ckpt["model_state_dict"])
    trainer.model.eval()

    logger.info("Running test evaluation...")
    test_ap, test_map = trainer._validate(which="test")
    logger.info(f"Test: ap_score={test_ap:.6f}  map_score={test_map:.6f}")

    out_path = Path("testMAPs") / (conf.task.name + ".txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"epoch_pt:       {script_args.epoch_pt}\n")
        f.write(f"best_epoch:     {best_epoch}\n")
        f.write(f"metric:         {metric}\n")
        f.write(f"delta:          {delta}\n")
        f.write(f"val_{metric}:   {best_val:.8f}\n")
        f.write(f"test_ap_score:  {test_ap:.8f}\n")
        f.write(f"test_map_score: {test_map:.8f}\n")
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
