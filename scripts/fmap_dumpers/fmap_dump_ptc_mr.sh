#!/usr/bin/env bash
# fmap_dump — dataset=ptc_mr
# --- CEAvg ---
python scripts/fmap_dump.py --dataset ptc_mr --fmap-loss ce-avg --no-fmap-symmetric --estol 1e-5

# --- CEMax ---
python scripts/fmap_dump.py --dataset ptc_mr --fmap-loss ce-adv-max --no-fmap-symmetric --estol 1e-5

echo 'Done.'
