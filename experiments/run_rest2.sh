#!/usr/bin/env bash
# Second queue: exploratory condition-matched normaliser, then the cost benchmark (single thread, core 3).
cd "$(dirname "$0")/.."
source /home/ali/Desktop/Articles/.venv/bin/activate
while kill -0 122078 2>/dev/null; do sleep 20; done
python experiments/run_condnorm.py --jobs 8 > results/logs/condnorm.log 2>&1
taskset -c 3 python experiments/cost.py > results/logs/cost.log 2>&1
echo "rest2 done" >> results/logs/rest2.log
