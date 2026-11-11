#!/usr/bin/env bash
# Full analysis pass over everything in results/ (no new scoring): calibration rules for every experiment,
# class-disjoint analysis, paired comparisons, compact tables, summary.md and figures. About 4 minutes.
set -eu
cd "$(dirname "$0")/.."
source /home/ali/Desktop/Articles/.venv/bin/activate
python experiments/summarize.py mswc samespk esc50 esc50_samesrc fleurs condnorm
python experiments/run_esc50_xclass.py --analyse
python experiments/paired.py
python experiments/paired_extra.py
python experiments/export_tables.py
python experiments/make_report.py > /dev/null
python experiments/figures.py
# Added after review: new variants, SSL baseline, masked same-speaker stream, review analyses.
python experiments/summarize.py mswc_x mswc_ssl mswc_ssl_lc samespk_v2 samespk_ssl
python experiments/review_analyses.py
echo "analysis done $(date)"
