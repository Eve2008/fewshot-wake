# Few-shot wake detection on Persian: code and results

`nexawake/features.py` and `nexawake/wake.py` are NexaRoom's deployed wake code (commit be63f3e), copied unchanged.
`nexawake/enroll.py` reproduces the deployed enrollment and calibration; `nexawake/fast.py` is a compiled copy of the
DTW matcher (equality with the deployed matcher: `tests/test_fast.py`).

## Run order

All scripts use the Python 3.12 venv of the workspace and at most 8 worker processes.

| step | script | output |
|---|---|---|
| tests | `python tests/run_tests.py` | compiled matcher = deployed matcher; chunk-parallel features and MFCC = single-stream |
| partition + bank | `experiments/build_bank.py` | `results/splits.json`, `results/bank/` (60-min bank, fp16 features) |
| cross-speaker MSWC | `experiments/run_mswc.py` | `results/mswc/tasks/*.json` (3 seeds x k = 3, 6, 10 x 16 words, 9 methods) |
| same speaker | `experiments/run_samespk.py` | `results/samespk/tasks/*.json` |
| ESC-50 sounds | `experiments/run_esc50.py` | `results/esc50/tasks/*.json` |
| ESC-50, same-source enrollment | `experiments/run_esc50_samesrc.py` | `results/esc50_samesrc/tasks/*.json`, `plan.json` |
| ESC-50, class-disjoint sound bank | `experiments/run_esc50_xclass.py`, then `--analyse` | `results/esc50_xclass/tasks/*.json`, `per_task.csv`, `calibration.csv`, `paired.csv` |
| noise diagnostics | `experiments/diagnose_noise.py`, `experiments/diagnose_fa_words.py` | `results/mswc/noise_only.csv`, `results/mswc/fa_words_seed1_k6_noaug.csv` |
| condition-matched normaliser (exploratory) | `experiments/run_condnorm.py` | `results/condnorm/tasks/*.json` |
| FLEURS out-of-domain negatives | `experiments/run_fleurs.py` | `results/fleurs/tasks/*.json` |
| compute cost | `taskset -c 19 python experiments/cost.py` (one pinned vCPU, one thread, 3 repeats, about 4 min) | `results/cost/` (`matcher_cost.csv`, `mfcc_dtw_cost.csv`, `bank_calibration_cost.csv`, `cost_info.json`, raw per-frame times) |
| manuscript checks | `experiments/manuscript_checks.py` (reads existing results only) | `results/manuscript_checks/` (orthographic variants of evaluation words; kinds of remaining false alarms) |
| after review: 2 x 2 ablation + logistic heads, cross-speaker | `experiments/run_extra.py` (variants in `experiments/methods_extra.py`; head negatives built once in `results/headneg/`) | `results/mswc_x/tasks/*.json` (same task names as `results/mswc/tasks`) |
| after review: same speaker with own clips masked from the seed-1 stream, all variants | `experiments/run_samespk_v2.py` | `results/samespk_v2/tasks/*.json` |
| after review: mHuBERT-147 + DTW baseline | `experiments/run_ssl_all.sh` (stages of `experiments/run_ssl.py`: layer selection on bank-side clips, bank + PCA, streams, task features, scoring) | `results/ssl/` (`layer_selection.json`, `pca.npz`, `tasks/*.json`) |
| after review: noise diagnostic on all 144 wakes, synthetic noise and recorded ESC-50 sounds | `experiments/diagnose_noise_v2.py` | `results/noise_v2/` |
| after review: original decision layer vs counting rule (seed 1, k = 6) | `experiments/run_detector_check.py` | `results/detector_check/` |
| after review: clips cut from sentences that contain the target | `experiments/check_sentence_overlap.py` | `results/manuscript_checks/sentence_overlap*.{csv,json}` |
| after review (second pass): same-speaker run resumed after a reboot (skips finished tasks) | `experiments/run_samespk_v2.py --jobs 6` | `results/samespk_v2/tasks/*.json` (325 built wakes + 7 unbuildable copies); log `results/logs/samespk_v2_resume.log` |
| after review: mHuBERT-147 + DTW on the same-speaker main set | `experiments/run_ssl_samespk.py` (feature pool uses `spawn`; see the comment in the script) | `results/ssl_samespk/` |
| after review: mHuBERT-147 + DTW with limited context (0.96 s past, at most 0.32 s look-ahead), seed 1 | `experiments/run_ssl_lc.py` | `results/ssl_lc/` |
| after review: size and cost of the mHuBERT-147 front end, offline and limited context | `experiments/cost_ssl.py`, `experiments/cost_ssl_lc.py` | `results/cost/ssl_cost.json`, `results/cost/ssl_lc_cost.json` |
| after review: both SSL runs in one queue | `experiments/run_review_queue2.sh` (stopped after the same-speaker run), then `experiments/run_ssl_lc_after.sh` (limited-context run with 4 processes) | logs `results/logs/ssl_samespk.log`, `ssl_lc.log`, `review_queue2.log` |
| after review: aggregates | `experiments/summarize.py mswc_x mswc_ssl mswc_ssl_lc samespk_v2 samespk_ssl` | `per_task.csv`, `calibration.csv`, `det.csv` in each result folder |
| after review: paired ablations, rule comparisons, held-out rule choice, tail models, own-voice alarms with intervals, Poisson reference | `experiments/review_analyses.py` (bootstrap generators seeded per comparison, so intervals do not depend on what else is computed) | `results/review/*.csv`, `results/review/summary_review.md` |

The log-odds logistic heads (`emb-head-lo`, `emb-head-aug-lo` in `methods_extra.py`) were scored on the same-speaker sets
only; the cross-speaker run of them was stopped at 6 of 144 tasks and nothing of it was merged into `results/mswc_x`. They
are not used in the manuscript.
| after review: analyses (paired 2 x 2 contrasts, tail vs order statistic, held-out rule choice, seed spread and two-way bootstrap, ML tail fits, tail-slope check) | `experiments/summarize.py mswc_x mswc_ssl samespk_v2`, then `experiments/review_analyses.py` | `results/review/*.csv`, `results/review/summary_review.md` |
| analysis, all at once | `experiments/run_analysis.sh` (about 3 min) | everything in the next row |
| analysis (in this order) | `experiments/summarize.py mswc samespk esc50 esc50_samesrc fleurs condnorm`, `experiments/run_esc50_xclass.py --analyse`, `experiments/paired.py`, `experiments/paired_extra.py`, `experiments/export_tables.py`, `experiments/make_report.py`, `experiments/figures.py` | `results/*/calibration.csv`, `per_task.csv`, `det.csv`, `paired*.csv`, `results/tables/*.csv`, `results/summary.md`, `results/figures/` |

`results/*/streams/` hold regenerable feature caches (large); they are not results.

## Data sources

- MLCommons Multilingual Spoken Words Corpus, Persian, test split (`data/mswc`).
- ESC-50 (Piczak 2015), read from the Hugging Face parquet mirror `ashraq/esc50` (extracted with
  `work/esc50-extract/extract.mjs`, hyparquet from npm, because pyarrow is not installable here).
- FLEURS Persian (`google/fleurs`, `fa_ir` test split) for out-of-domain negatives.
- openWakeWord's `melspectrogram.onnx` and `embedding_model.onnx` feature models (`data/models`).

The pilot (k = 6, 0.625 h negatives, older script) is kept in `experiments/pilot/` and `results/pilot/`.

## Large derived tables

`results/samespk/per_task.csv` and `results/samespk_v2/per_task.csv` (65-84 MB) are not committed. Regenerate them from the committed `tasks/*.json` with `python experiments/summarize.py samespk`. Feature caches (`*.npy`) are also regenerated by the experiment scripts.
