#!/usr/bin/env bash
# Queue of the smaller post-review runs: waits for the decision-layer check, then the noise diagnostic and the
# sentence-overlap control; waits for the masked same-speaker run, then the same-speaker SSL baseline.
set -u
cd "$(dirname "$0")/.."
source /home/ali/Desktop/Articles/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
running() { ps -eo args | awk -v s="$1" '$1 ~ /python/ && index($0, s) {found=1} END {exit !found}'; }
while running "run_detector_check.py"; do sleep 20; done
python experiments/diagnose_noise_v2.py --jobs 3 > results/logs/noise_v2.log 2>&1
python experiments/check_sentence_overlap.py --jobs 3 > results/logs/sentence_overlap.log 2>&1
while running "run_samespk_v2.py"; do sleep 20; done
SSL_THREADS=2 python experiments/run_ssl_samespk.py --procs 3 > results/logs/ssl_samespk.log 2>&1
echo "queue done $(date)"
