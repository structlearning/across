"""Generate and save AugNet random projection hyperplanes.

Creates greform_hyperplanes_short_K.pt and greform_hyperplanes_long_K.pt
in allPklDumps/, where K = --num-hyperplanes.

  Short form shape: (K, 1 + filters_3)
  Long  form shape: (K, 2 + filters_3)

Usage
-----
python -B scripts/generate_hyperplanes.py --num-hyperplanes 4
python -B scripts/generate_hyperplanes.py --num-hyperplanes 8 --filters-3 10
"""

import argparse
import os
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Generate AugNet random projection hyperplanes."
    )
    parser.add_argument("--num-hyperplanes", type=int, required=True,
                        help="Number of random hyperplanes K (aug_embedder.num_hyperplanes)")
    parser.add_argument("--filters-3", type=int, default=10,
                        help="GNN last-layer output dimension d1 (gmn.filters_3, default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Torch manual seed (default: 42)")
    parser.add_argument("--out-dir", type=str, default="allPklDumps",
                        help="Output directory (default: allPklDumps)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    K, d1 = args.num_hyperplanes, args.filters_3

    for form, cols in [("short", 1 + d1), ("long", 2 + d1)]:
        torch.manual_seed(args.seed)
        hyperplanes = torch.randn(K, cols)
        out_path = out_dir / f"greform_hyperplanes_{form}_{K}.pt"
        torch.save(hyperplanes, out_path)
        print(f"Saved {form:5s} hyperplanes: shape={tuple(hyperplanes.shape)}  -> {out_path}")


if __name__ == "__main__":
    main()
