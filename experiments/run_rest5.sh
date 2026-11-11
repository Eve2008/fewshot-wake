#!/usr/bin/env bash
# Part 2 queue, run next to the ESC-50 run (6 workers) so that we use about 8 processes in total:
# pure-noise diagnostic, FLEURS out-of-domain negatives, condition-matched normaliser (2 workers each).
# Every runner skips outputs that already exist, so the queue can be restarted safely.
set -u
cd "$(dirname "$0")/.."
source /home/ali/Desktop/Articles/.venv/bin/activate
[ -f results/mswc/noise_only.csv ] || python experiments/diagnose_noise.py --jobs 2 > results/logs/noise_only.log 2>&1
python experiments/run_fleurs.py --jobs 2 > results/logs/fleurs.log 2>&1
python experiments/run_condnorm.py --jobs 2 > results/logs/condnorm.log 2>&1
echo "rest5 done $(date)" >> results/logs/rest5.log
