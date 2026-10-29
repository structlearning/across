"""
GPU-aware command queue runner.

Reads a file of shell commands (one per line) and dispatches them to GPUs as
memory becomes available. CUDA_VISIBLE_DEVICES= prefixes and trailing & are
stripped from each command; the assigned GPU is injected via the subprocess
environment instead.

Blank lines and lines starting with # are ignored. The script is resumable:
state is written to a JSON file next to the commands file, so re-running the
same invocation picks up where it left off.

Usage:
    python -m reform.runner.cmd_queue <commands_file> [options]

Options:
    --min-free-mb INT    Minimum free GPU memory in MiB to consider a GPU
                         available (default: 10000)
    --poll-interval INT  Seconds between GPU polls (default: 60)
    --max-retries INT    Maximum retries for a failed command (default: 3)
    --max-jobs INT       Maximum concurrent jobs (default: unlimited)
    --state-file PATH    Explicit path for state JSON (default: auto-derived
                         from commands file name + content hash)

Example:
    python -m reform.runner.cmd_queue my_runs.txt --min-free-mb 12000
"""

# ruff: noqa: E402
from dotenv import load_dotenv

load_dotenv()

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

from loguru import logger


# ---------------------------------------------------------------------------
# Command parsing
# ---------------------------------------------------------------------------

_CUDA_PREFIX_RE = re.compile(r"^(CUDA_VISIBLE_DEVICES=\S+\s+)+")


def _parse_commands(path: Path) -> list[str]:
    """Read commands file, skip blanks and comments, return cleaned commands."""
    raw_lines = path.read_text().splitlines()
    commands: list[str] = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Strip CUDA_VISIBLE_DEVICES=X prefix(es)
        stripped = _CUDA_PREFIX_RE.sub("", stripped)
        # Strip trailing &
        stripped = stripped.rstrip("& ").rstrip()
        if stripped:
            commands.append(stripped)
    return commands


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class CmdQueueOrchestrator:
    def __init__(
        self,
        commands: list[str],
        state_file: Path,
        min_free_mb: int,
        poll_interval: int,
        max_retries: int,
        max_jobs: int | None = None,
        no_gpu_manage: bool = False,
        retry_failed: bool = False,
        dispatch_delay: float = 0,
        name: str | None = None,
        allowed_gpus: set[int] | None = None,
        one_job_per_gpu: bool = False,
    ):
        self.commands = commands
        self.state_file = state_file
        self.min_free_mb = min_free_mb
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.max_jobs = max_jobs
        self.no_gpu_manage = no_gpu_manage
        self.retry_failed = retry_failed
        self.dispatch_delay = dispatch_delay
        self.name = name
        self.allowed_gpus = allowed_gpus
        self.one_job_per_gpu = one_job_per_gpu
        # run_key -> Popen, for live processes in this session
        self._processes: dict[str, subprocess.Popen] = {}

    def _log(self, level: str, msg: str) -> None:
        prefix = f"[{self.name}] " if self.name else ""
        getattr(logger, level)(f"{prefix}{msg}")

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run_key(idx: int) -> str:
        return f"cmd_{idx:04d}"

    def _load_state(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file) as f:
                state = json.load(f)
            self._log("info", f"Resumed state from {self.state_file}")
            for key, info in state.items():
                if info["status"] == "running":
                    self._log("warning",
                        f"{key}: was 'running' in state file with no live process"
                        " — marking pending"
                    )
                    info["status"] = "pending"
                elif info["status"] == "failed" and self.retry_failed:
                    self._log("info", f"{key}: retrying previously failed command")
                    info["status"] = "pending"
                    info["retries"] = 0
            return state

        state = {
            self._run_key(idx): {
                "cmd": cmd,
                "status": "pending",
                "retries": 0,
            }
            for idx, cmd in enumerate(self.commands)
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
            self._log("warning", f"nvidia-smi failed: {e}")
            return []

        candidates: list[tuple[int, int]] = []
        for line in result.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) != 2:
                continue
            idx, free_mib = int(parts[0].strip()), int(parts[1].strip())
            if self.allowed_gpus is not None and idx not in self.allowed_gpus:
                continue
            if free_mib >= self.min_free_mb:
                candidates.append((idx, free_mib))

        candidates.sort(key=lambda x: -x[1])  # most free first
        return candidates

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, run_key: str, cmd: str, gpu_id: int | None) -> subprocess.Popen:
        if gpu_id is not None:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
            gpu_label = f"GPU {gpu_id}"
        else:
            env = None
            gpu_label = "inherited GPU"
        proc = subprocess.Popen(cmd, shell=True, env=env, start_new_session=True)
        self._log("info", f"Dispatched {run_key} → {gpu_label} (pid={proc.pid})\n  cmd: {cmd}")
        return proc

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _check_processes(self, state: dict) -> dict:
        for run_key, proc in list(self._processes.items()):
            ret = proc.poll()
            if ret is None:
                continue

            del self._processes[run_key]

            if ret == 0:
                state[run_key]["status"] = "done"
                self._log("info", f"{run_key} completed successfully")
            else:
                state[run_key]["retries"] += 1
                retries = state[run_key]["retries"]
                if retries > self.max_retries:
                    state[run_key]["status"] = "failed"
                    self._log("error",
                        f"{run_key} failed permanently after {retries} attempt(s) (exit {ret})"
                    )
                else:
                    state[run_key]["status"] = "pending"
                    self._log("warning",
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

                self._log("info",
                    f"Status — done: {len(by_status['done'])}, "
                    f"running: {len(by_status['running'])}, "
                    f"pending: {len(by_status['pending'])}, "
                    f"failed: {len(by_status['failed'])}"
                )

                if not by_status["pending"] and not by_status["running"]:
                    self._log("info", "All commands finished.")
                    if by_status["failed"]:
                        self._log("warning", f"Failed commands: {by_status['failed']}")
                    break

                if by_status["pending"]:
                    if self.max_jobs is not None and len(by_status["running"]) >= self.max_jobs:
                        self._log("info",
                            f"Max jobs ({self.max_jobs}) running. Waiting {self.poll_interval}s..."
                        )
                    elif self.no_gpu_manage:
                        for run_key in by_status["pending"]:
                            if self.max_jobs is not None and len(self._processes) >= self.max_jobs:
                                break
                            cmd = state[run_key]["cmd"]
                            proc = self._dispatch(run_key, cmd, gpu_id=None)
                            state[run_key].update({"status": "running", "pid": proc.pid})
                            self._processes[run_key] = proc
                            self._save_state(state)
                            if self.dispatch_delay > 0:
                                time.sleep(self.dispatch_delay)
                    else:
                        # Subtract min_free_mb per already-dispatched run to avoid
                        # over-committing before processes actually allocate memory.
                        free_mib: dict[int, int] = dict(self._available_gpus())

                        if self.one_job_per_gpu:
                            busy = {
                                info.get("gpu")
                                for info in state.values()
                                if info["status"] == "running" and "gpu" in info
                            }
                            free_mib = {k: v for k, v in free_mib.items() if k not in busy}

                        if not free_mib:
                            self._log("info",
                                f"No GPUs with >= {self.min_free_mb} MiB free right now. "
                                f"Waiting {self.poll_interval}s..."
                            )
                        else:
                            for run_key in by_status["pending"]:
                                if self.max_jobs is not None and len(self._processes) >= self.max_jobs:
                                    break
                                eligible = [
                                    (idx, mib) for idx, mib in free_mib.items()
                                    if mib >= self.min_free_mb
                                ]
                                if not eligible:
                                    break
                                gpu_id, _ = max(eligible, key=lambda x: x[1])
                                if self.one_job_per_gpu:
                                    del free_mib[gpu_id]
                                else:
                                    free_mib[gpu_id] -= self.min_free_mb
                                cmd = state[run_key]["cmd"]
                                proc = self._dispatch(run_key, cmd, gpu_id)
                                state[run_key].update(
                                    {"status": "running", "gpu": gpu_id, "pid": proc.pid}
                                )
                                self._processes[run_key] = proc
                                self._save_state(state)
                                if self.dispatch_delay > 0:
                                    time.sleep(self.dispatch_delay)

                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            self._log("warning", "Interrupted — terminating running subprocesses...")
            for run_key, proc in self._processes.items():
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                state[run_key]["status"] = "pending"
            self._save_state(state)
            self._log("info", f"State saved to {self.state_file}. Re-run to resume.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU-aware command queue runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("commands_file", type=Path, help="File with one shell command per line")
    parser.add_argument("--name", type=str, default=None,
                        help="Human-readable identifier for this queue (shown in all log output)")
    parser.add_argument("--min-free-mb", type=int, default=10000,
                        help="Minimum free GPU memory (MiB) to dispatch a command")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between GPU polls")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retries for a failed command")
    parser.add_argument("--max-jobs", type=int, default=None,
                        help="Maximum number of concurrently running jobs (default: unlimited)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU indices to use (e.g. 0,1,2,3). "
                             "By default all GPUs are eligible.")
    parser.add_argument("--one-job-per-gpu", action="store_true",
                        help="Allow at most one running job per GPU (instead of packing by memory).")
    parser.add_argument("--no-gpu-manage", action="store_true",
                        help="Skip nvidia-smi and CUDA_VISIBLE_DEVICES injection (use for SLURM/MIG)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-queue all failed commands from a previous run")
    parser.add_argument("--dispatch-delay", type=float, default=0,
                        help="Seconds to sleep between dispatching each job (to avoid API rate limits)")
    parser.add_argument("--state-file", type=Path, default=None,
                        help="Exact path for the state JSON file (auto-derived if omitted)")
    parser.add_argument("--state-dir", type=Path, default=None,
                        help="Directory to write the auto-named state file into (created if needed)")
    args = parser.parse_args()

    if args.state_file and args.state_dir:
        parser.error("--state-file and --state-dir are mutually exclusive")

    commands_file: Path = args.commands_file.resolve()
    if not commands_file.exists():
        parser.error(f"Commands file not found: {commands_file}")

    commands = _parse_commands(commands_file)
    if not commands:
        logger.warning("No commands found in file. Exiting.")
        return

    name: str | None = args.name
    prefix = f"[{name}] " if name else ""
    logger.info(f"{prefix}Loaded {len(commands)} command(s) from {commands_file}")

    if args.state_file is not None:
        state_file = args.state_file
    else:
        content_hash = hashlib.sha256(commands_file.read_bytes()).hexdigest()[:8]
        stem = name if name else commands_file.stem
        state_dir = args.state_dir.resolve() if args.state_dir else commands_file.parent
        state_dir.mkdir(parents=True, exist_ok=True)
        state_file = state_dir / f"{stem}_state_{content_hash}.json"

    logger.info(f"{prefix}State file: {state_file}")

    allowed_gpus = (
        {int(x.strip()) for x in args.gpus.split(",")} if args.gpus else None
    )
    if allowed_gpus is not None:
        logger.info(f"{prefix}Restricting to GPUs: {sorted(allowed_gpus)}")

    orchestrator = CmdQueueOrchestrator(
        commands=commands,
        state_file=state_file,
        min_free_mb=args.min_free_mb,
        poll_interval=args.poll_interval,
        max_retries=args.max_retries,
        max_jobs=args.max_jobs,
        no_gpu_manage=args.no_gpu_manage,
        retry_failed=args.retry_failed,
        dispatch_delay=args.dispatch_delay,
        name=name,
        allowed_gpus=allowed_gpus,
        one_job_per_gpu=args.one_job_per_gpu,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
