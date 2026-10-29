# ruff: noqa: E402
from dotenv import load_dotenv

# Load environment variables before importing torch
load_dotenv()

import argparse
import hashlib
import json
import operator
import os
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger

# # ── project root on path ──────────────────────────────────────────────────────
# ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, ROOT)
# os.chdir(ROOT)

from reform.lsh.constants import (
    LAMBDA_COLL_MIN_VALUES,
    LAMBDA_FENCE_SIT,
    LINEAR_EVERY_DIM_VALUES,
    MAX_RETRIES,
    MIN_FREE_MB,
    POLL_INTERVAL,
)
from reform.utils.script_utils import (
    CORPUS_SIZE, add_aug_args, _build_ckpt_tag, _build_aug_ckpt_tag,
)
from scripts.misc.wandb_to_human import convert_name

# Usage Example:
#   python -m reform.runner.lsh_sweep \
#       model.name=GEN dataset.name=ptc_fr dataset.rel_mode=sub_iso \
#       lsh.dump_dir=allPklDumps/... \
#       training.batch_size=8192 training.loss='ce-avg' \
#       aug_embedder.aug_form=long aug_embedder.num_hyperplanes=4 \
#       aug_embedder.push_minus=False aug_embedder.symmetric=True
#       aug_embedder.model_every_dim=False
#
# The orchestrator writes state to lsh_sweep_state.json in the current directory.
# If interrupted, re-running the same command will resume from where it left off.


class SweepOrchestrator:
    def __init__(
        self,
        lambda_values: list[float],
        linear_every_dim_values: list[bool],
        base_overrides: list[str],
        state_file: Path,
        min_free_mb: int = MIN_FREE_MB,
        poll_interval: int = POLL_INTERVAL,
        max_retries: int = MAX_RETRIES,
        retry_failed: bool = False,
    ):
        self.lambda_values = lambda_values
        self.linear_every_dim_values = linear_every_dim_values
        self.base_overrides = base_overrides
        self.state_file = state_file
        self.min_free_mb = min_free_mb
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.retry_failed = retry_failed
        # run_key -> Popen, for live processes in this session
        self._processes: dict[str, subprocess.Popen] = {}

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run_key(lambda_coll_min: float, linear_every_dim: bool) -> str:
        suffix = "linperd" if linear_every_dim else "nolinperd"
        return f"lambda_{lambda_coll_min:.2f}_{suffix}"

    def _load_state(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file) as f:
                state = json.load(f)
            logger.info(f"Resumed state from {self.state_file}")
            # Runs that were 'running' when the script last died are now orphaned.
            for key, info in state.items():
                if info["status"] == "running":
                    logger.warning(
                        f"{key}: was 'running' in state file with no live process"
                        " — marking pending"
                    )
                    info["status"] = "pending"
                elif info["status"] == "failed" and self.retry_failed:
                    logger.warning(f"{key}: --retry-failed set — resetting to pending")
                    info["status"] = "pending"
                    info["retries"] = 0
            return state

        state = {
            self._run_key(lam, linperd): {
                "lambda_coll_min": lam,
                "linear_every_dim": linperd,
                "status": "pending",
                "retries": 0,
            }
            for lam in self.lambda_values
            for linperd in self.linear_every_dim_values
        }
        self._save_state(state)
        return state

    def _save_state(self, state: dict) -> None:
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    # ------------------------------------------------------------------
    # GPU helpers
    # ------------------------------------------------------------------

    def _available_gpus(self) -> list[tuple[int, int]]:
        """Return (gpu_idx, free_mib) pairs for GPUs with >= min_free_mb free,
        sorted by free memory descending."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logger.warning(f"nvidia-smi failed: {e}")
            return []

        candidates: list[tuple[int, int]] = []
        for line in result.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) != 2:
                continue
            idx, free_mib = int(parts[0].strip()), int(parts[1].strip())
            if free_mib >= self.min_free_mb:
                candidates.append((idx, free_mib))

        candidates.sort(key=lambda x: -x[1])  # most free first
        return candidates

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self, run_key: str, lambda_coll_min: float, linear_every_dim: bool, gpu_id: int
    ) -> subprocess.Popen:
        lambda_bit_bal = round(1.0 - LAMBDA_FENCE_SIT - lambda_coll_min, 10)
        sweep_overrides = [
            f"lsh.lambda_fence_sit={LAMBDA_FENCE_SIT}",
            f"lsh.lambda_coll_min={lambda_coll_min}",
            f"lsh.lambda_bit_bal={lambda_bit_bal}",
            f"lsh.linear_every_dim={linear_every_dim}",
            "task.wandb_project=across_lsh_train",
            "task.wandb_mode=offline",
        ]
        cmd = [sys.executable, "-m", "reform.runner.lsh_run", *self.base_overrides, *sweep_overrides]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
        proc = subprocess.Popen(cmd, env=env)
        logger.info(f"Dispatched {run_key} → GPU {gpu_id} (pid={proc.pid})")
        return proc

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _check_processes(self, state: dict) -> dict:
        """Poll live processes; update state for anything that finished."""
        for run_key, proc in list(self._processes.items()):
            ret = proc.poll()
            if ret is None:
                continue  # still running

            del self._processes[run_key]

            if ret == 0:
                state[run_key]["status"] = "done"
                logger.info(f"{run_key} completed successfully")
            else:
                state[run_key]["retries"] += 1
                retries = state[run_key]["retries"]
                if retries > self.max_retries:
                    state[run_key]["status"] = "failed"
                    logger.error(
                        f"{run_key} failed permanently after {retries} attempt(s) (exit {ret})"
                    )
                else:
                    state[run_key]["status"] = "pending"
                    logger.warning(
                        f"{run_key} failed (exit {ret}), queued for retry"
                        f" {retries}/{self.max_retries}"
                    )
        return state

    def run(self) -> None:
        state = self._load_state()

        try:
            while True:
                state = self._check_processes(state)
                self._save_state(state)

                by_status: dict[str, list[str]] = {
                    "pending": [], "running": [], "done": [], "failed": [],
                }
                for key, info in state.items():
                    by_status.setdefault(info["status"], []).append(key)

                logger.info(
                    f"Status — done: {len(by_status['done'])}, "
                    f"running: {len(by_status['running'])}, "
                    f"pending: {len(by_status['pending'])}, "
                    f"failed: {len(by_status['failed'])}"
                )

                if not by_status["pending"] and not by_status["running"]:
                    logger.info("All runs finished.")
                    if by_status["failed"]:
                        logger.warning(f"Failed runs: {by_status['failed']}")
                    break

                if by_status["pending"]:
                    # free_mib tracks remaining estimated memory per GPU,
                    # subtracting min_free_mb per dispatched run to avoid
                    # over-committing before processes actually allocate memory.
                    free_mib: dict[int, int] = dict(self._available_gpus())

                    if not free_mib:
                        logger.info(
                            f"No GPUs with >= {self.min_free_mb} MiB free right now. "
                            f"Waiting {self.poll_interval}s..."
                        )
                    else:
                        for run_key in by_status["pending"]:
                            # Pick GPU with most estimated free memory
                            eligible = [
                                (idx, mib) for idx, mib in free_mib.items()
                                if mib >= self.min_free_mb
                            ]
                            if not eligible:
                                break
                            gpu_id, _ = max(eligible, key=operator.itemgetter(1))
                            free_mib[gpu_id] -= self.min_free_mb
                            lam = state[run_key]["lambda_coll_min"]
                            linperd = state[run_key]["linear_every_dim"]
                            proc = self._dispatch(run_key, lam, linperd, gpu_id)
                            state[run_key].update(
                                {"status": "running", "gpu": gpu_id, "pid": proc.pid}
                            )
                            self._processes[run_key] = proc
                            self._save_state(state)

                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            logger.warning("Interrupted — terminating running subprocesses...")
            for run_key, proc in self._processes.items():
                proc.terminate()
                state[run_key]["status"] = "pending"  # re-queue on next run
            self._save_state(state)
            logger.info(f"State saved to {self.state_file}. Re-run the command to resume.")


def _build_name_string(overrides: list[str]) -> str:
    """Build a synthetic name string that convert_name can parse."""
    params: dict[str, str] = {}
    for o in overrides:
        if "=" in o:
            k, v = o.split("=", 1)
            params[k] = v

    loss = params.get("training.loss", "")
    sym = params.get("aug_embedder.symmetric", "True").lower() == "true"
    model_every_dim = params.get("aug_embedder.model_every_dim", "False").lower() == "true"
    push_minus = params.get("aug_embedder.push_minus", "False").lower() == "true"
    aug_form = params.get("aug_embedder.aug_form", "")
    linear_every_dim = params.get("lsh.linear_every_dim", "False").lower() == "true"

    parts = [f"loss{loss}", "_sym_" if sym else "_asym_"]
    if model_every_dim:
        parts.append("everydim")
    if push_minus:
        parts.append("pushminus")
    parts.append(f"_af{aug_form}")
    if linear_every_dim:
        parts.append("linperd")

    return "".join(parts)


def dump_sweep_csv(
    csv_path: Path,
    base_overrides: list[str],
    lambda_values: list[float],
    linear_every_dim_values: list[bool],
) -> None:
    import csv
    import socket

    machine = socket.gethostname()

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Run Readable Name", "Machine",
            "Collision Minimizer Lambda", "Bit Balance Lambda",
        ])
        for linperd in linear_every_dim_values:
            overrides_with_linperd = [*base_overrides, f"lsh.linear_every_dim={linperd}"]
            readable_name = convert_name(_build_name_string(overrides_with_linperd))
            for lam in lambda_values:
                lambda_bit_bal = round(1.0 - LAMBDA_FENCE_SIT - lam, 10)
                writer.writerow([readable_name, machine, lam, lambda_bit_bal])

    logger.info(f"Sweep plan written to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate LSH training sweep across all lambda/linear_every_dim combinations."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g. ptc_fr)")
    parser.add_argument("--retry-failed", action="store_true", default=False,
                        help="Re-queue any 'failed' entries in the state file as pending.")
    parser.add_argument("--third-loss-type", type=str, default="sc_coll_min",
                        choices=["sc_coll_min", "ad_decorr"],
                        help="Third loss type for LSH training (default: sc_coll_min)")
    add_aug_args(parser)
    args = parser.parse_args()

    ckpt_tag     = _build_ckpt_tag(args)
    aug_ckpt_tag = _build_aug_ckpt_tag(args)

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
        "training.batch_size=8192",
        "aug_embedder.model=mlp",
        f"aug_embedder.symmetric={args.aug_symmetric}",
        f"aug_embedder.hidden_dim={args.aug_hidden_dim}",
        f"aug_embedder.num_layers={args.aug_num_layers}",
        f"aug_embedder.output_dim={args.aug_output_dim}",
        f"aug_embedder.aug_form={args.aug_form}",
        f"aug_embedder.model_every_dim={args.aug_model_every_dim}",
        f"aug_embedder.push_minus={args.aug_push_minus}",
        # e.g. bestValidationModels/GEN_ptc_fr_numC=100000_MinR=0.05_MaxR=0.25_scoring_function=cos,stemp=0.1,margin=1.0/best.pt
        f"aug_embedder.best_checkpoint_path=bestValidationModels/{ckpt_tag}/best.pt",
        # e.g. latestModels/GEN_ptc_fr_numC=100000_MinR=0.05_MaxR=0.25_scoring_function=cos,stemp=0.1,margin=1.0
        f"aug_embedder.latest_ckpts_path=latestModels/{ckpt_tag}",
        # e.g. bestValidationModels/GEN_ptc_fr_numC=100000_..._aug_emb_mlp_sym_h64_l2_o10_lossce-adv-max_afshort_pushminus_aug_corpus_initfix_pretrained_embs/best.pt
        f"lsh.aug_checkpoint_path=bestValidationModels/{aug_ckpt_tag}/best.pt",
        f"lsh.third_loss_type={args.third_loss_type}",
        "lsh.dump_dir=./allPklDumps",
        "aug_embedder.use_pretrained_embs=True",
        "dataset.data_type=idx",
        "use_cluster=0",
        f"early_stopping.tolerance={args.estol}",
        *([] if args.num_epochs is None else [f"training.num_epochs={args.num_epochs}"]),
    ]

    config_hash = hashlib.sha256(" ".join(base_overrides).encode()).hexdigest()[:8]
    state_file = Path(f"./allPklDumps/lsh_sweep_state_{config_hash}.json")

    dump_sweep_csv(
        csv_path=Path(f"./allPklDumps/lsh_sweep_runs_{config_hash}.csv"),
        base_overrides=base_overrides,
        lambda_values=LAMBDA_COLL_MIN_VALUES,
        linear_every_dim_values=LINEAR_EVERY_DIM_VALUES,
    )

    orchestrator = SweepOrchestrator(
        lambda_values=LAMBDA_COLL_MIN_VALUES,
        linear_every_dim_values=LINEAR_EVERY_DIM_VALUES,
        base_overrides=base_overrides,
        state_file=state_file,
        min_free_mb=MIN_FREE_MB,
        poll_interval=POLL_INTERVAL,
        max_retries=MAX_RETRIES,
        retry_failed=args.retry_failed,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
