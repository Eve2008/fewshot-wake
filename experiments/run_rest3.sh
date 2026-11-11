#!/usr/bin/env bash
# Third queue: FLEURS out-of-domain negatives (after the second queue).
cd "$(dirname "$0")/.."
source /home/ali/Desktop/Articles/.venv/bin/activate
while kill -0 165891 2>/dev/null; do sleep 20; done
python experiments/run_fleurs.py --jobs 8 > results/logs/fleurs.log 2>&1
echo "rest3 done" >> results/logs/rest3.log
