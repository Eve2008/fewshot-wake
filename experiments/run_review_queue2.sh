#!/usr/bin/env bash
# Second post-review queue (after the reboot): same-speaker SSL baseline, then the limited-context SSL control.
set -u
cd "$(dirname "$0")/.."
source /home/ali/Desktop/Articles/.venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SSL_THREADS=2
python experiments/run_ssl_samespk.py --procs 3 > results/logs/ssl_samespk.log 2>&1
echo "ssl_samespk exit $?" >> results/logs/review_queue2.log
python experiments/run_ssl_lc.py --procs 3 > results/logs/ssl_lc.log 2>&1
echo "ssl_lc exit $? $(date)" >> results/logs/review_queue2.log
