"""Download the release bundle from Hugging Face Hub and verify sha256.

Usage
-----
    python -B scripts/reproduce/download_assets.py
    python -B scripts/reproduce/download_assets.py --repo pritishc/across-neurips2026 --revision v1.0
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_REPO = "pritishc/across-neurips2026"
DEFAULT_REVISION = "v1.0"


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HF dataset repo id.")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Git tag or commit SHA.")
    parser.add_argument("--dest", default=str(REPO_ROOT), help="Download target directory.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip sha256 verification.")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is not installed. Run `pip install huggingface_hub` first.", file=sys.stderr)
        sys.exit(1)

    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo}@{args.revision} to {dest}")
    local = snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(dest),
    )
    local = Path(local)
    print(f"Downloaded to {local}")

    if args.skip_verify:
        print("Skipping sha256 verification per --skip-verify.")
        return

    manifest_path = local / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: manifest.json missing at {manifest_path}", file=sys.stderr)
        sys.exit(1)
    manifest = json.loads(manifest_path.read_text())

    bad = []
    checked = 0
    for section in ("tasks", "data"):
        for entry in manifest.get(section, []):
            paths_and_hashes = []
            if section == "tasks":
                if entry.get("status") != "ok":
                    continue
                paths_and_hashes.append((entry["ckpt"], entry["ckpt_sha256"]))
                paths_and_hashes.append((entry["log"], entry["log_sha256"]))
            else:
                paths_and_hashes.append((entry["path"], entry["sha256"]))
            for rel, expected in paths_and_hashes:
                checked += 1
                full = local / rel
                if not full.exists():
                    bad.append((rel, "missing"))
                    continue
                actual = sha256_of(full)
                if actual != expected:
                    bad.append((rel, f"sha256 mismatch: expected {expected[:12]}... got {actual[:12]}..."))

    print(f"\nVerified {checked} files.")
    if bad:
        print(f"{len(bad)} failures:")
        for rel, reason in bad[:10]:
            print(f"  {rel}: {reason}")
        if len(bad) > 10:
            print(f"  ... and {len(bad) - 10} more")
        sys.exit(1)
    print("All files verified.")


if __name__ == "__main__":
    main()
