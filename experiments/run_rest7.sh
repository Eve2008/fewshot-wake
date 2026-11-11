#!/usr/bin/env bash
# ESC-50 at k = 3 in two passes so that the cheap methods finish first: once every k = 6 task file
# exists, stop the combined run (it has started k = 3 tasks with all methods), score k = 3 without
# MFCC-DTW, then add MFCC-DTW to the same task files (run_esc50.py merges missing methods).
set -u
cd "$(dirname "$0")/.."
source /home/ali/Desktop/Articles/.venv/bin/activate
while [ "$(ls results/esc50/tasks/*_k6.json 2>/dev/null | wc -l)" -lt 150 ] && pgrep -f "experiments/run_esc50.py --jobs 6$" >/dev/null; do sleep 15; done
pkill -f "experiments/run_esc50.py --jobs 6$"
sleep 5
pkill -9 -f "experiments/run_esc50.py --jobs 6$"
echo "k6 complete, combined run stopped $(date)" >> results/logs/rest7.log
python experiments/run_esc50.py --jobs 6 --shots 3 --methods nexa,nexa-min,nexa-noaug,nexa-noaug-bnorm,emb-only,mel-only,prototype > results/logs/esc50_k3.log 2>&1
echo "k3 without MFCC done $(date)" >> results/logs/rest7.log
python experiments/run_esc50.py --jobs 6 --shots 3 --methods mfcc-dtw > results/logs/esc50_k3_mfcc.log 2>&1
echo "rest7 done $(date)" >> results/logs/rest7.log
