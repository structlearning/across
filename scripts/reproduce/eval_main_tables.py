"""Reproduce the paper's tables from the released checkpoint bundle.

Iterates `scripts/reproduce/tasks.jsonl`, calls `scripts/eval_at_epoch.py`
per task, then renders the LaTeX tables.

Usage
-----
    python -B scripts/reproduce/eval_main_tables.py
    python -B scripts/reproduce/eval_main_tables.py --tables 1 6 7 8 9
    python -B scripts/reproduce/eval_main_tables.py --jobs 4
    python -B scripts/reproduce/eval_main_tables.py --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_FILE = REPO_ROOT / "scripts" / "reproduce" / "tasks.jsonl"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_at_epoch.py"
TESTMAPS = REPO_ROOT / "testMAPs"
OUTPUT_DIR = REPO_ROOT / "output" / "tables"

EPOCH_PT = 200
METRIC = "ap_score"
# Eval uses the config-default early_stopping.tolerance (5e-3). This is NOT the
# `estol=1e-5` invariant that AugNet pipeline commands require.

COMMON_OVERRIDES = [
    "model.name=GEN",
    "dataset.rel_mode=sub_iso",
    "training.batch_size=128",
    'model.scoring_function="hinge"',
    "training.loss=celoss",
    "training.stratified_batching=False",
    "early_stopping.patience=200",
    "early_stopping.metric=ap_score",
    "task.wandb_enabled=false",
]

# (script, output stem, extra args). --appendix auto-adds `_appendix` to stem.
TABLE_RENDERERS = {
    1:       ("scripts/latex/make_results_table.py",  "results_table",  []),
    2:       ("scripts/latex/make_augnet_table.py",   "augnet_table",   []),
    3:       ("scripts/latex/make_fmap_table.py",     "fmap_table",     []),
    4:       ("scripts/latex/make_ablation_table.py", "ablation_table", []),
    "6-9":   ("scripts/latex/make_results_table.py",  "results_table",  ["--appendix"]),
    "18-21": ("scripts/latex/make_augnet_table.py",   "augnet_table",   ["--appendix"]),
    "22-25": ("scripts/latex/make_fmap_table.py",     "fmap_table",     ["--appendix"]),
    "26-29": ("scripts/latex/make_ablation_table.py", "ablation_table", ["--appendix"]),
}


def load_tasks(filter_tables: set[int] | None = None) -> list[dict]:
    tasks = [json.loads(line) for line in TASKS_FILE.read_text().splitlines() if line.strip()]
    if filter_tables:
        tasks = [t for t in tasks if any(p in filter_tables for p in t["paper_tables"])]
    seen = {}
    for t in tasks:
        key = (t["task_name"], t.get("augnet"))
        if key not in seen:
            seen[key] = t
    return list(seen.values())


def overrides_for(task: dict) -> list[str]:
    overrides = list(COMMON_OVERRIDES)
    overrides.append(f"dataset.name={task['dataset']}")
    overrides.append(f"use_cluster={task['cluster']}")
    overrides.append(f"training.relationship_load={task['rel']}")
    if task.get("suffix"):
        overrides.append(f"cluster_suffix={task['suffix']}")
    if task.get("augnet"):
        overrides.append(f"cluster_rel_subdir={task['augnet']}")
    return overrides


def run_one(task: dict, dry_run: bool = False) -> tuple[dict, int, str]:
    overrides = overrides_for(task)
    cmd = [
        sys.executable, "-B", str(EVAL_SCRIPT),
        "--epoch_pt", str(EPOCH_PT),
        "--metric", METRIC,
        *overrides,
    ]
    if dry_run:
        return task, 0, " ".join(cmd)
    res = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    log = res.stdout + "\n" + res.stderr
    return task, res.returncode, log


def label(task: dict) -> str:
    aug = task.get("augnet") or "-"
    return f"{task['dataset']}/cl{task['cluster']}/{task['rel']}/{aug[:50]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", nargs="+", type=int, default=None,
                        help="Restrict to a subset of paper tables.")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Parallel eval workers.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print eval commands without running them.")
    parser.add_argument("--dump-cmds", type=str, default=None,
                        help="Write one eval command per line to this file "
                             "(for scripts/cmd_queue.py) and exit.")
    parser.add_argument("--skip-render", action="store_true",
                        help="Skip the final table-rendering step.")
    args = parser.parse_args()

    filter_set = set(args.tables) if args.tables else None
    tasks = load_tasks(filter_set)
    print(f"Loaded {len(tasks)} unique tasks" + (f" (filtered to tables {sorted(filter_set)})" if filter_set else ""))

    if args.dump_cmds:
        out = Path(args.dump_cmds)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for t in tasks:
                parts = [
                    "python", "-B", "scripts/eval_at_epoch.py",
                    "--epoch_pt", str(EPOCH_PT), "--metric", METRIC,
                    *overrides_for(t),
                ]
                f.write(" ".join(parts) + "\n")
        print(f"Wrote {len(tasks)} commands to {out}")
        return

    failures = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, t, args.dry_run): t for t in tasks}
        for fut in as_completed(futures):
            task, rc, log = fut.result()
            completed += 1
            if args.dry_run:
                print(f"[dry-run {completed}/{len(tasks)}] {label(task)}")
                print(f"  {log}")
                continue
            tag = "ok " if rc == 0 else "FAIL"
            print(f"[{tag} {completed}/{len(tasks)}] {label(task)}")
            if rc != 0:
                failures.append((task, log))

    if args.dry_run:
        return

    print(f"\n{len(tasks) - len(failures)}/{len(tasks)} eval tasks succeeded.")
    if failures:
        print(f"\n{len(failures)} failures:")
        for task, log in failures[:5]:
            print(f"  {label(task)}:\n    {log.strip().splitlines()[-5:]}")
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more")

    if args.skip_render or failures:
        return

    print("\nRendering LaTeX tables...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    which_tables = (
        sorted(filter_set) if filter_set
        else [1, 2, 3, 4, "6-9", "18-21", "22-25", "26-29"]
    )
    for key in which_tables:
        entry = TABLE_RENDERERS.get(key)
        if not entry:
            continue
        script, stem, extra = entry
        out = OUTPUT_DIR / f"{stem}.tex"
        cmd = [sys.executable, "-B", script, "--out", str(out), *extra]
        print(f"  $ {' '.join(cmd)}")
        subprocess.run(cmd, cwd=REPO_ROOT, check=False)


if __name__ == "__main__":
    main()
