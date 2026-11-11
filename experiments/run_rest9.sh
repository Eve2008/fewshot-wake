#!/usr/bin/env bash
# Replaces the tail of rest8: once the k = 3 ESC-50 passes (rest7) are done, restart the same-source
# MFCC-DTW fill with 6 workers (it was running with 2) and finish the class-disjoint scoring with 2.
set -u
cd "$(dirname "$0")/.."
source /home/ali/Desktop/Articles/.venv/bin/activate
while pgrep -f "experiments/run_rest7.sh" >/dev/null; do sleep 20; done
pkill -f "experiments/run_esc50_samesrc.py --jobs 2$"
sleep 5
pkill -9 -f "experiments/run_esc50_samesrc.py --jobs 2$"
python experiments/run_esc50_xclass.py --jobs 2 --shots 6,3 > results/logs/esc50_xclass2.log 2>&1 &
python experiments/run_esc50_samesrc.py --jobs 6 > results/logs/esc50_samesrc_mfcc2.log 2>&1
wait
echo "rest9 done $(date)" >> results/logs/rest9.log
