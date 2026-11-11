#!/usr/bin/env bash
# Runs the remaining experiments in sequence after the main MSWC run (8 worker processes each).
set -u
cd "$(dirname "$0")/.."
source /home/ali/Desktop/Articles/.venv/bin/activate
while pgrep -f "experiments/run_mswc.py" >/dev/null; do sleep 20; done
python experiments/run_samespk.py --jobs 8 > results/logs/samespk.log 2>&1
python experiments/run_esc50.py --jobs 8 > results/logs/esc50.log 2>&1
python experiments/diagnose_noise.py > results/logs/noise_only.log 2>&1
echo "rest done" >> results/logs/rest.log
