#!/usr/bin/env bash
# Part 2, third side queue (2 workers next to the 6-worker ESC-50 run): same-source ESC-50 wakes
# without MFCC-DTW, then MFCC-DTW added; after the ESC-50 run has finished (rest7), the class-disjoint
# scoring of the remaining k = 6 and all k = 3 wakes. Existing outputs are skipped or completed.
# NOTE: stopped during the MFCC step (2 workers were too slow); run_rest9.sh finishes both steps.
set -u
cd "$(dirname "$0")/.."
source /home/ali/Desktop/Articles/.venv/bin/activate
while pgrep -f "experiments/run_rest6.sh" >/dev/null; do sleep 20; done
python experiments/run_esc50_samesrc.py --jobs 2 --methods nexa,nexa-min,nexa-noaug,nexa-noaug-bnorm,emb-only,prototype > results/logs/esc50_samesrc.log 2>&1
echo "samesrc without MFCC done $(date)" >> results/logs/rest8.log
python experiments/run_esc50_samesrc.py --jobs 2 > results/logs/esc50_samesrc_mfcc.log 2>&1
echo "samesrc MFCC done $(date)" >> results/logs/rest8.log
while pgrep -f "experiments/run_rest7.sh" >/dev/null; do sleep 20; done
python experiments/run_esc50_xclass.py --jobs 8 --shots 6,3 > results/logs/esc50_xclass2.log 2>&1
echo "rest8 done $(date)" >> results/logs/rest8.log
