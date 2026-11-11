#!/usr/bin/env bash
# Part 2, second side queue: after rest5 (noise, FLEURS, condnorm), build the class-disjoint ESC-50
# streams and score the wakes whose main ESC-50 task file already exists (2 workers, next to the
# 6-worker ESC-50 run). Rerun after the ESC-50 run ends to finish the rest; existing files are skipped.
set -u
cd "$(dirname "$0")/.."
source /home/ali/Desktop/Articles/.venv/bin/activate
while pgrep -f "experiments/run_rest5.sh" >/dev/null; do sleep 20; done
python experiments/run_esc50_xclass.py --jobs 2 --shots 6 > results/logs/esc50_xclass.log 2>&1
echo "rest6 done $(date)" >> results/logs/rest6.log
