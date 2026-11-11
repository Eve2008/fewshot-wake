#!/usr/bin/env bash
# Waits for the same-speaker SSL run, then runs the limited-context SSL control with 4 processes x 2 threads.
set -u
cd "$(dirname "$0")/.."
source /home/ali/Desktop/Articles/.venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SSL_THREADS=2
while pgrep -f "run_ssl_samespk.py" > /dev/null; do sleep 10; done
echo "ssl_samespk finished $(date)" >> results/logs/review_queue2.log
python experiments/run_ssl_lc.py --procs 4 > results/logs/ssl_lc.log 2>&1
echo "ssl_lc exit $? $(date)" >> results/logs/review_queue2.log
