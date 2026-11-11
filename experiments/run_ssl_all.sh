#!/usr/bin/env bash
# SSL baseline in resumable stages: seed 1 first (features of positives and enrollment, then scoring), then seeds 2-3.
set -eu
cd "$(dirname "$0")/.."
source /home/ali/Desktop/Articles/.venv/bin/activate
export SSL_THREADS=2
python experiments/run_ssl.py --procs 3 --seeds 1 --steps tasks
python experiments/run_ssl.py --procs 3 --seeds 1 --steps score
python experiments/run_ssl.py --procs 3 --seeds 2,3 --steps streams,tasks
python experiments/run_ssl.py --procs 3 --seeds 2,3 --steps score
echo "ssl all done $(date)"
