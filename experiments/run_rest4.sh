#!/usr/bin/env bash
# Resume queue (the earlier queues were interrupted): after the same-speaker run, run
# ESC-50, the pure-noise diagnostic, the condition-matched normaliser, the FLEURS
# negatives, and finally the single-thread cost benchmark with nothing else of ours running.
# Every runner skips task files that already exist, so the queue can be restarted safely.
set -u
cd "$(dirname "$0")/.."
source /home/ali/Desktop/Articles/.venv/bin/activate
while pgrep -f "experiments/run_samespk.py" >/dev/null; do sleep 20; done
python experiments/run_esc50.py --jobs 8 > results/logs/esc50.log 2>&1
python experiments/diagnose_noise.py > results/logs/noise_only.log 2>&1
python experiments/run_condnorm.py --jobs 8 > results/logs/condnorm.log 2>&1
python experiments/run_fleurs.py --jobs 8 > results/logs/fleurs.log 2>&1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 3 python experiments/cost.py > results/logs/cost.log 2>&1
echo "rest4 done $(date)" >> results/logs/rest4.log
