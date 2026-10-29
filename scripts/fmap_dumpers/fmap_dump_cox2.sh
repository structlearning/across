#!/usr/bin/env bash
# fmap_dump — dataset=cox2
# --- CEAvg ---
python scripts/fmap_dump.py --dataset cox2 --fmap-loss ce-avg --no-fmap-symmetric --estol 1e-5

# --- CEMax ---
python scripts/fmap_dump.py --dataset cox2 --fmap-loss ce-adv-max --no-fmap-symmetric --estol 1e-5

echo 'Done.'
