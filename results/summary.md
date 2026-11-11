# Few-shot wake detection: results of the step-1 experiments

All numbers below were produced by the scripts in `experiments/` on this machine; raw per-task results are in `results/*/tasks/*.json`, per-wake rows in `results/*/per_task.csv`, aggregates with 95 % cluster-bootstrap intervals (resampling words, speakers or classes) in `results/*/calibration.csv`.

## Data and partition

- Persian MSWC test split: 120,232 one-second clips, 3,197 speakers, 12,231 words.
- Evaluation words never used as negatives or in the bank: 34 distinct words (the 16 cross-speaker targets and the 25 words that some speaker repeats >= 8 times; 7 words are on both lists).
- Bank: 2,880 clips (60.0 min with 0.25 s gaps) from 610 bank-only speakers and 1282 bank-only words; 45,002 feature frames = 11.52 MB at fp16 (noisy copy another 11.52 MB; MFCC copy for the baseline 28.08 MB).
- Evaluation negatives per seed: 11,520 clips = seed 1: 4.000 h, seed 2: 4.000 h, seed 3: 4.000 h (clean; the same audio with white+pink noise at 10 dB SNR as the noisy condition). They share no speaker and no word with the bank and contain none of the 34 evaluation words; the three seeds use different clips (speakers and words do recur across seeds). Room audio for the deployed calibration: 25.2 s per seed.
- FA/h resolution: one false alarm on a 4 h stream is 0.25 FA/h for that wake; per k the 48 cross-speaker wakes are scored on 192 wake-hours of clean negatives (12.0 h of distinct audio over the three seeds).
- Near-miss words: the evaluation negatives exclude the target words themselves, but 83-88 of the 11,520 clips per seed are other words that contain a target as a substring (e.g. دوستان for دوست); they are kept as negatives.
- Cross-speaker positives: 60 recordings by 60 speakers not enrolled, per word and seed (16 words x 3 seeds = 48 wakes per k, 2,880 positives per k). Enrollment uses k recordings by k different speakers; nested across k (k = 3 is a subset of k = 6, which is a subset of k = 10). The deployed enrollment rejected a recording in 0 of 48 wakes at k = 3, 3 of 48 wakes at k = 6, 3 of 48 wakes at k = 10 (the wake was still built from the remaining recordings).

## 0. Main findings (numbers from the tables below)

1. **The deployed calibration (20 s of room audio) is far too loose.** Cross-speaker, clean: the deployed matcher reaches a mean of 116 to 131 FA/h over k = 3, 6, 10 (median 77 to 90), the clean-template variant 97 to 111 FA/h; at most 4% of the 48 wakes stay within 2 FA/h.
2. **Per-wake calibration on the 60-min bank (11.5 MB at fp16) with a tail fit lands near the target on disjoint negatives.** Clean templates, target 1 FA/h: mean 0.71 to 0.79 FA/h over k (median 0.50 to 0.62, worst wake 3.25 to 4.00), 90% to 96% of wakes within 2 FA/h, at FRR 0.819 to 0.875. Targets 0.5 and 2 FA/h give 0.24 to 0.30 and 1.78 to 2.02 FA/h. The deployed matcher with the same rule: 1.03 to 1.27 FA/h at target 1.
3. **The naive empirical rule overshoots by design.** Taking the largest threshold whose bank rate is still <= 1 FA/h allows one bank event in 1 h and gives 1.94 to 2.12 FA/h; the order-statistic rule (no bank event allowed) gives 0.80 to 1.08 FA/h but cannot distinguish targets below one event per bank length (0.5 and 1 FA/h give the same threshold); the tail fit can.
4. **One global threshold learned on the bank vs. per-wake bank calibration.** Clean templates, k = 6, target 1: global FRR 0.860, mean 1.01 FA/h, worst wake 14.50 FA/h; per-wake tail fit FRR 0.828, mean 0.79, worst wake 3.25. For the deployed matcher the global threshold's worst wake reaches 42.75 FA/h (per-wake: 4.50). Paired over the 48 wakes, per-wake minus global: dFRR -0.032 [-0.084, +0.024], dFA/h -0.21 [-1.15, +0.61]; neither interval excludes 0.
5. **Method comparison at matched FA/h (each wake's own 1 FA/h threshold on the evaluation negatives), cross-speaker.** Dropping the noise-augmented template copies lowers FRR from 0.937 to 0.946 to 0.770 to 0.827 (paired dFRR -0.167 to -0.117; 3 of 3 intervals entirely below 0). Without the background normaliser FRR is 0.960 to 0.966 (deployed minus no-normaliser -0.029 to -0.015). Fitting the normaliser on the bank instead of the room audio changes FRR by -0.010 to -0.002. Clean templates beat the 96-d embedding alone by 0.070 to 0.117, MFCC + DTW by 0.126 to 0.154, the mean-embedding prototype by 0.143 to 0.203 and the 32-d log-mel part alone by 0.171 to 0.229 in FRR. Even the best variant misses 76% to 82% of cross-speaker positives at 1 FA/h.
6. **Effect of the number of enrollment shots.** k = 10 minus k = 3 at matched FA/h: clean templates dFRR -0.057 [-0.083, -0.036] (k = 10 minus 6: -0.016 [-0.033, -0.000]); deployed matcher -0.008 [-0.026, +0.010].
7. **Thresholds set on clean audio do not transfer to 10 dB noise that appears after enrollment.** k = 6: the deployed matcher (noise-augmented templates) gives 2080 FA/h with the deployed calibration and 700 FA/h with the bank tail fit; clean templates stay quiet (0.01 FA/h) but miss 99% of positives.
8. **Same-speaker enrollment (MSWC speakers who repeat a word, main set, k = 10, 66 speakers).** Clean templates with the bank tail fit: FRR 0.652 [0.587, 0.717] at 0.91 FA/h on other speakers but 5.48 FA/h on the same speaker's other words (4.4 h pooled); with the deployed calibration the same templates give FRR 0.223 at 100 FA/h. The deployed matcher with the bank tail fit: FRR 0.797. MFCC + DTW reaches FRR 0.448 at 1.16 FA/h on other speakers but 56.6 FA/h on the speaker's own other words: it responds to the voice, not only to the word. Broad set (k = 3, 130 speakers): clean templates FRR 0.700 at 0.85 FA/h (own voice 4.77).
9. **The bank thresholds stay conservative on out-of-domain continuous speech.** On 3.76 h of FLEURS Persian read speech the 48 cross-speaker wakes (k = 6) with clean templates and the bank tail fit give 0.006 FA/h (worst wake 0.27); the deployed calibration gives 29.5 FA/h (deployed matcher) and 9.2 FA/h (clean templates). Isolated words are the harder negative material.
10. **The noise-augmented templates match noise itself.** On 10 min of pure white + pink noise (seed 1, k = 6, clean-bank threshold at 1 FA/h) the deployed matcher fires 781 times per hour on average (median 6, max 2940); clean templates fire 0 times. Normalising every stream with the statistics of its own condition (exploratory, seed 1) removes the false-alarm burst in noise (0.05 and 0.05 FA/h with the bank tail fit) but not the misses (FRR 1.000 and 0.994; 0.909 even with each wake's own threshold on the noisy negatives).
11. **Few-shot sound wakes (ESC-50, k = 6, 150 wakes from 50 classes, cross-recording) are not usable at 1 FA/h against dense environmental sound.** The deployed calibration gives 460 FA/h (FRR 0.179); a speech-only bank cannot calibrate a sound wake (clean templates, tail fit: 200 FA/h over speech and sounds together; 2.2 FA/h on the speech stream, 664 FA/h on the sound stream). Adding 1.17 h of ESC-50 folds 1-2 (other classes) to the bank meets the target (0.70 FA/h) but FRR is 0.968; even each wake's own threshold on the evaluation negatives leaves FRR >= 0.931 (best method: MFCC + DTW). At 10 FA/h (own threshold) the best FRR is 0.779 (MFCC + DTW).
12. **Same-source enrollment and unseen sound types.** Enrolling 3 takes of the same recording and testing on its other takes (114 wakes): clean templates with the speech + sound bank reach FRR 0.769 [0.662, 0.865] at 0.82 FA/h; the best own-threshold FRR is 0.520 (MFCC + DTW). A sound bank that lacks the evaluation sound types (class-disjoint halves, clean templates, k = 6) raises the sound-stream FA/h from 2.80 to 3.32 (paired +0.52 [+0.13, +0.97]) at identical FRR.
13. **Compute cost is small next to the 80 ms frame (single thread, x86 virtual machine; Raspberry Pi not measured).** Feature extraction takes 3.06 ms per frame. With 5-frame templates the deployed NumPy matcher and decision layer add 0.23 ms for 6 clean templates and 0.24 ms for 18 (k = 6 with the two noise copies), 0.38 ms for 240; the compiled matcher needs 0.008 ms for 18. Features + matcher for one k = 6 wake use 4.1 % of real time. Calibrating one wake on the 60-min bank takes 0.20 s (6 templates, compiled matcher) or 10.6 s with the deployed NumPy code. The MFCC + DTW baseline (6 templates of 34 10-ms frames) costs 0.14 ms per 80 ms of audio (compiled DTW; batch MFCC 0.07 ms), against 3.07 ms for features + compiled matcher with 6 clean templates.

## 1. Calibration against the shipped negative bank (cross-speaker, clean)

**deployed matcher (noise-augmented templates), k = 6, target 1 FA/h, 48 wakes**

| threshold rule | FRR [95% CI] | FA/h mean [95% CI] | FA/h median | FA/h p90 | FA/h worst wake | share of wakes <= 2 FA/h |
|---|---|---|---|---|---|---|
| deployed (20 s room audio) | 0.586 [0.503, 0.666] | 117.89 [86.72, 153.16] | 79.125 | 334.9 | 434.0 | 0.000 |
| one global threshold (bank, other wakes) | 0.975 [0.955, 0.989] | 2.59 [0.82, 4.87] | 0.250 | 5.225 | 42.750 | 0.854 |
| bank 15 min, empirical | 0.897 [0.861, 0.928] | 5.57 [3.56, 7.79] | 2.625 | 11.500 | 33.250 | 0.458 |
| bank 30 min, empirical | 0.927 [0.898, 0.952] | 2.60 [1.91, 3.37] | 1.750 | 7.000 | 9.250 | 0.625 |
| bank 60 min, empirical | 0.920 [0.887, 0.949] | 2.57 [2.02, 3.10] | 2.000 | 5.075 | 7.000 | 0.521 |
| bank 60 min, order statistic | 0.950 [0.930, 0.968] | 1.15 [0.86, 1.45] | 0.750 | 2.400 | 5.000 | 0.875 |
| bank 30 min, tail fit | 0.963 [0.943, 0.980] | 0.79 [0.50, 1.09] | 0.500 | 1.825 | 4.500 | 0.958 |
| bank 60 min, tail fit | 0.947 [0.918, 0.969] | 1.15 [0.83, 1.51] | 0.750 | 2.575 | 4.500 | 0.812 |
| oracle global (eval negatives) | 0.986 [0.974, 0.994] | 0.86 [0.33, 1.43] | 0.000 | 2.775 | 8.500 | 0.875 |
| oracle per wake (eval negatives) | 0.946 [0.926, 0.965] | 0.92 [0.87, 0.95] | 1.000 | 1.000 | 1.000 | 1.000 |

**clean templates only, k = 6, target 1 FA/h, 48 wakes**

| threshold rule | FRR [95% CI] | FA/h mean [95% CI] | FA/h median | FA/h p90 | FA/h worst wake | share of wakes <= 2 FA/h |
|---|---|---|---|---|---|---|
| deployed (20 s room audio) | 0.277 [0.217, 0.340] | 106.21 [80.19, 136.00] | 84.375 | 226.9 | 383.2 | 0.000 |
| one global threshold (bank, other wakes) | 0.860 [0.808, 0.911] | 1.01 [0.31, 1.89] | 0.250 | 1.575 | 14.500 | 0.938 |
| bank 15 min, empirical | 0.740 [0.682, 0.800] | 3.15 [1.90, 4.90] | 1.875 | 6.675 | 21.250 | 0.562 |
| bank 30 min, empirical | 0.778 [0.732, 0.824] | 1.78 [1.32, 2.30] | 1.375 | 3.225 | 9.500 | 0.771 |
| bank 60 min, empirical | 0.745 [0.695, 0.794] | 2.12 [1.54, 2.85] | 1.625 | 3.900 | 9.500 | 0.667 |
| bank 60 min, order statistic | 0.820 [0.781, 0.861] | 0.90 [0.71, 1.15] | 0.750 | 1.825 | 5.250 | 0.958 |
| bank 30 min, tail fit | 0.862 [0.830, 0.892] | 0.51 [0.35, 0.69] | 0.250 | 1.500 | 3.000 | 0.958 |
| bank 60 min, tail fit | 0.828 [0.795, 0.861] | 0.79 [0.57, 1.03] | 0.625 | 1.900 | 3.250 | 0.896 |
| oracle global (eval negatives) | 0.852 [0.798, 0.899] | 0.94 [0.41, 1.61] | 0.250 | 1.575 | 10.750 | 0.938 |
| oracle per wake (eval negatives) | 0.786 [0.748, 0.828] | 0.96 [0.94, 0.98] | 1.000 | 1.000 | 1.000 | 1.000 |

**Achieved FA/h on the evaluation negatives vs. target (clean templates; columns = target, k)**

| rule | 0.5 FA/h, k=3 | 0.5 FA/h, k=6 | 0.5 FA/h, k=10 | 1 FA/h, k=3 | 1 FA/h, k=6 | 1 FA/h, k=10 | 2 FA/h, k=3 | 2 FA/h, k=6 | 2 FA/h, k=10 |
|---|---|---|---|---|---|---|---|---|---|
| bank 60 min, empirical | 1.083 | 0.896 | 0.797 | 1.943 | 2.120 | 1.964 | 3.062 | 3.182 | 3.250 |
| bank 60 min, order statistic | 1.083 | 0.896 | 0.797 | 1.083 | 0.896 | 0.797 | 1.943 | 2.120 | 1.964 |
| bank 60 min, tail fit | 0.297 | 0.260 | 0.245 | 0.708 | 0.792 | 0.755 | 1.839 | 2.016 | 1.781 |
| one global threshold (bank, other wakes) | 0.682 | 0.505 | 0.448 | 1.109 | 1.005 | 1.057 | 2.177 | 2.260 | 2.286 |
| oracle per wake (eval negatives) | 0.474 | 0.474 | 0.479 | 0.984 | 0.964 | 0.937 | 1.901 | 1.891 | 1.937 |

**FRR at the same thresholds (clean templates; columns = target, k)**

| rule | 0.5 FA/h, k=3 | 0.5 FA/h, k=6 | 0.5 FA/h, k=10 | 1 FA/h, k=3 | 1 FA/h, k=6 | 1 FA/h, k=10 | 2 FA/h, k=3 | 2 FA/h, k=6 | 2 FA/h, k=10 |
|---|---|---|---|---|---|---|---|---|---|
| bank 60 min, empirical | 0.864 | 0.820 | 0.818 | 0.809 | 0.745 | 0.727 | 0.750 | 0.690 | 0.674 |
| bank 60 min, order statistic | 0.864 | 0.820 | 0.818 | 0.864 | 0.820 | 0.818 | 0.809 | 0.745 | 0.727 |
| bank 60 min, tail fit | 0.928 | 0.899 | 0.890 | 0.875 | 0.828 | 0.819 | 0.803 | 0.739 | 0.731 |
| one global threshold (bank, other wakes) | 0.912 | 0.905 | 0.887 | 0.880 | 0.860 | 0.830 | 0.831 | 0.798 | 0.775 |
| oracle per wake (eval negatives) | 0.865 | 0.841 | 0.816 | 0.827 | 0.786 | 0.770 | 0.780 | 0.720 | 0.699 |

**Spread of the achieved FA/h over wakes (cross-speaker, clean, target 1 FA/h; 48 wakes per row)**

| method | rule | k | FRR | FA/h mean | median | p90 | worst wake | share <= 2 FA/h |
|---|---|---|---|---|---|---|---|---|
| clean templates only | bank 60 min, empirical | 3 | 0.809 | 1.943 | 1.375 | 3.900 | 8.000 | 0.646 |
| clean templates only | bank 60 min, empirical | 6 | 0.745 | 2.120 | 1.625 | 3.900 | 9.500 | 0.667 |
| clean templates only | bank 60 min, empirical | 10 | 0.727 | 1.964 | 1.750 | 3.900 | 6.250 | 0.604 |
| clean templates only | bank 60 min, order statistic | 3 | 0.864 | 1.083 | 0.500 | 2.900 | 5.750 | 0.812 |
| clean templates only | bank 60 min, order statistic | 6 | 0.820 | 0.896 | 0.750 | 1.825 | 5.250 | 0.958 |
| clean templates only | bank 60 min, order statistic | 10 | 0.818 | 0.797 | 0.500 | 1.750 | 5.500 | 0.917 |
| clean templates only | bank 60 min, tail fit | 3 | 0.875 | 0.708 | 0.500 | 1.500 | 4.000 | 0.938 |
| clean templates only | bank 60 min, tail fit | 6 | 0.828 | 0.792 | 0.625 | 1.900 | 3.250 | 0.896 |
| clean templates only | bank 60 min, tail fit | 10 | 0.819 | 0.755 | 0.500 | 1.750 | 3.750 | 0.958 |
| clean templates only | one global threshold (bank, other wakes) | 3 | 0.880 | 1.109 | 0.250 | 2.075 | 14.250 | 0.896 |
| clean templates only | one global threshold (bank, other wakes) | 6 | 0.860 | 1.005 | 0.250 | 1.575 | 14.500 | 0.938 |
| clean templates only | one global threshold (bank, other wakes) | 10 | 0.830 | 1.057 | 0.375 | 3.300 | 8.250 | 0.875 |
| clean templates only | oracle global (eval negatives) | 3 | 0.877 | 0.979 | 0.250 | 2.500 | 10.500 | 0.854 |
| clean templates only | oracle global (eval negatives) | 6 | 0.852 | 0.943 | 0.250 | 1.575 | 10.750 | 0.938 |
| clean templates only | oracle global (eval negatives) | 10 | 0.829 | 0.995 | 0.500 | 3.000 | 8.250 | 0.875 |
| deployed matcher (noise-augmented templates) | bank 60 min, empirical | 3 | 0.912 | 2.901 | 2.375 | 5.975 | 8.750 | 0.438 |
| deployed matcher (noise-augmented templates) | bank 60 min, empirical | 6 | 0.920 | 2.568 | 2.000 | 5.075 | 7.000 | 0.521 |
| deployed matcher (noise-augmented templates) | bank 60 min, empirical | 10 | 0.918 | 2.547 | 1.750 | 6.400 | 8.250 | 0.562 |
| deployed matcher (noise-augmented templates) | bank 60 min, order statistic | 3 | 0.948 | 1.375 | 0.875 | 3.000 | 6.750 | 0.750 |
| deployed matcher (noise-augmented templates) | bank 60 min, order statistic | 6 | 0.950 | 1.151 | 0.750 | 2.400 | 5.000 | 0.875 |
| deployed matcher (noise-augmented templates) | bank 60 min, order statistic | 10 | 0.952 | 1.104 | 0.625 | 3.150 | 5.750 | 0.854 |
| deployed matcher (noise-augmented templates) | bank 60 min, tail fit | 3 | 0.950 | 1.271 | 0.875 | 2.800 | 6.000 | 0.875 |
| deployed matcher (noise-augmented templates) | bank 60 min, tail fit | 6 | 0.947 | 1.151 | 0.750 | 2.575 | 4.500 | 0.812 |
| deployed matcher (noise-augmented templates) | bank 60 min, tail fit | 10 | 0.945 | 1.031 | 0.750 | 2.500 | 4.250 | 0.812 |
| deployed matcher (noise-augmented templates) | one global threshold (bank, other wakes) | 3 | 0.965 | 2.646 | 0.375 | 4.750 | 37.000 | 0.854 |
| deployed matcher (noise-augmented templates) | one global threshold (bank, other wakes) | 6 | 0.975 | 2.594 | 0.250 | 5.225 | 42.750 | 0.854 |
| deployed matcher (noise-augmented templates) | one global threshold (bank, other wakes) | 10 | 0.966 | 2.120 | 0.250 | 8.500 | 19.000 | 0.854 |
| deployed matcher (noise-augmented templates) | oracle global (eval negatives) | 3 | 0.984 | 1.000 | 0.250 | 1.800 | 17.750 | 0.896 |
| deployed matcher (noise-augmented templates) | oracle global (eval negatives) | 6 | 0.986 | 0.865 | 0.000 | 2.775 | 8.500 | 0.875 |
| deployed matcher (noise-augmented templates) | oracle global (eval negatives) | 10 | 0.980 | 0.969 | 0.000 | 4.100 | 9.000 | 0.854 |

**Tail-fit sensitivity to the fitted event range (clean templates, k = 6, target 1 FA/h)**

| fit range (bank events) | FRR | FA/h mean | FA/h median | share of wakes <= 2 FA/h |
|---|---|---|---|---|
| [2, 50] (default) | 0.828 | 0.792 | 0.625 | 0.896 |
| [1,20] | 0.814 | 0.984 | 0.750 | 0.896 |
| [2,100] | 0.840 | 0.672 | 0.500 | 0.917 |
| [3,30] | 0.829 | 0.776 | 0.625 | 0.896 |
| [5,200] | 0.870 | 0.406 | 0.250 | 0.979 |

**Paired differences, bank 60 min tail fit minus another rule (same wakes; cross-speaker, clean, target 1 FA/h; 95 % bootstrap over words)**

| method | k | compared with | wakes | dFRR [95% CI] | dFA/h [95% CI] |
|---|---|---|---|---|---|
| deployed matcher (noise-augmented templates) | 3 | bank 60 min, empirical | 48 | +0.038 [+0.026, +0.050] | -1.63 [-2.07, -1.25] |
| deployed matcher (noise-augmented templates) | 6 | bank 60 min, empirical | 48 | +0.026 [+0.018, +0.035] | -1.42 [-1.80, -1.06] |
| deployed matcher (noise-augmented templates) | 10 | bank 60 min, empirical | 48 | +0.027 [+0.018, +0.039] | -1.52 [-2.05, -0.99] |
| deployed matcher (noise-augmented templates) | 3 | deployed (20 s room audio) | 48 | +0.371 [+0.306, +0.435] | -114.45 [-148.72, -85.81] |
| deployed matcher (noise-augmented templates) | 6 | deployed (20 s room audio) | 48 | +0.361 [+0.280, +0.444] | -116.74 [-150.01, -84.29] |
| deployed matcher (noise-augmented templates) | 10 | deployed (20 s room audio) | 48 | +0.370 [+0.292, +0.445] | -130.34 [-165.24, -96.66] |
| deployed matcher (noise-augmented templates) | 3 | one global threshold (bank, other wakes) | 48 | -0.015 [-0.038, +0.004] | -1.37 [-3.56, +0.42] |
| deployed matcher (noise-augmented templates) | 6 | one global threshold (bank, other wakes) | 48 | -0.028 [-0.058, -0.002] | -1.44 [-3.60, +0.30] |
| deployed matcher (noise-augmented templates) | 10 | one global threshold (bank, other wakes) | 48 | -0.020 [-0.050, +0.007] | -1.09 [-3.33, +0.39] |
| deployed matcher (noise-augmented templates) | 3 | oracle per wake (eval negatives) | 48 | +0.005 [-0.003, +0.015] | +0.32 [-0.02, +0.76] |
| deployed matcher (noise-augmented templates) | 6 | oracle per wake (eval negatives) | 48 | +0.001 [-0.011, +0.013] | +0.23 [-0.08, +0.58] |
| deployed matcher (noise-augmented templates) | 10 | oracle per wake (eval negatives) | 48 | +0.009 [-0.006, +0.022] | +0.14 [-0.12, +0.41] |
| clean templates only | 3 | bank 60 min, empirical | 48 | +0.066 [+0.048, +0.086] | -1.23 [-1.54, -0.92] |
| clean templates only | 6 | bank 60 min, empirical | 48 | +0.083 [+0.065, +0.103] | -1.33 [-1.84, -0.91] |
| clean templates only | 10 | bank 60 min, empirical | 48 | +0.092 [+0.074, +0.111] | -1.21 [-1.47, -0.93] |
| clean templates only | 3 | deployed (20 s room audio) | 48 | +0.527 [+0.468, +0.578] | -96.66 [-128.88, -68.18] |
| clean templates only | 6 | deployed (20 s room audio) | 48 | +0.552 [+0.486, +0.616] | -105.42 [-135.01, -78.13] |
| clean templates only | 10 | deployed (20 s room audio) | 48 | +0.545 [+0.484, +0.609] | -110.15 [-137.55, -84.44] |
| clean templates only | 3 | one global threshold (bank, other wakes) | 48 | -0.006 [-0.043, +0.041] | -0.40 [-1.41, +0.46] |
| clean templates only | 6 | one global threshold (bank, other wakes) | 48 | -0.032 [-0.084, +0.024] | -0.21 [-1.15, +0.61] |
| clean templates only | 10 | one global threshold (bank, other wakes) | 48 | -0.011 [-0.067, +0.048] | -0.30 [-1.34, +0.48] |
| clean templates only | 3 | oracle per wake (eval negatives) | 48 | +0.047 [+0.026, +0.070] | -0.28 [-0.55, +0.05] |
| clean templates only | 6 | oracle per wake (eval negatives) | 48 | +0.042 [+0.016, +0.070] | -0.17 [-0.39, +0.07] |
| clean templates only | 10 | oracle per wake (eval negatives) | 48 | +0.049 [+0.016, +0.085] | -0.18 [-0.41, +0.08] |

## 2. Methods and enrollment size (cross-speaker, clean, target 1 FA/h)

FRR (achieved FA/h) at the deployable bank 60 min tail-fit threshold; FRR at the oracle global threshold and at each wake's own oracle threshold (eval negatives) as references. 48 wakes per cell.

| method | k=3 FRR (FA/h) | k=3 oracle-global FRR | k=3 oracle/wake FRR | k=6 FRR (FA/h) | k=6 oracle-global FRR | k=6 oracle/wake FRR | k=10 FRR (FA/h) | k=10 oracle-global FRR | k=10 oracle/wake FRR |
|---|---|---|---|---|---|---|---|---|---|
| deployed matcher (noise-augmented templates) | 0.950 (1.27) | 0.984 | 0.945 | 0.947 (1.15) | 0.986 | 0.946 | 0.945 (1.03) | 0.980 | 0.937 |
| deployed, best single template | 0.933 (1.13) | 0.969 | 0.925 | 0.939 (0.98) | 0.970 | 0.934 | 0.939 (0.89) | 0.972 | 0.930 |
| clean templates only | 0.875 (0.71) | 0.877 | 0.827 | 0.828 (0.79) | 0.852 | 0.786 | 0.819 (0.76) | 0.829 | 0.770 |
| clean templates, bank normaliser | 0.872 (0.70) | 0.878 | 0.825 | 0.828 (0.77) | 0.847 | 0.778 | 0.823 (0.65) | 0.833 | 0.760 |
| deployed, no normaliser | 0.966 (0.76) | 0.998 | 0.960 | 0.973 (0.66) | 0.997 | 0.965 | 0.977 (0.74) | 0.997 | 0.966 |
| embedding part only (96-d) | 0.928 (0.64) | 0.936 | 0.897 | 0.941 (0.55) | 0.963 | 0.903 | 0.930 (0.59) | 0.947 | 0.887 |
| log-mel part only (32-d) | 0.998 (2.46) | 0.999 | 0.999 | 0.998 (2.96) | 0.999 | 0.999 | 0.998 (2.90) | 0.999 | 0.999 |
| prototype (mean embedding) | 0.977 (1.01) | 0.983 | 0.970 | 0.977 (0.94) | 0.980 | 0.970 | 0.978 (1.05) | 0.986 | 0.973 |
| MFCC + DTW | 0.966 (0.92) | 0.991 | 0.953 | 0.948 (0.69) | 0.978 | 0.924 | 0.944 (0.64) | 0.982 | 0.924 |

Deployed calibration (20 s room audio) for comparison:

| method | k=3 FRR / FA/h | k=6 FRR / FA/h | k=10 FRR / FA/h |
|---|---|---|---|
| deployed matcher (noise-augmented templates) | 0.579 / 115.7 | 0.586 / 117.9 | 0.575 / 131.4 |
| clean templates only | 0.348 / 97.4 | 0.277 / 106.2 | 0.274 / 110.9 |
| deployed, no normaliser | 0.665 / 95.0 | 0.649 / 98.2 | 0.644 / 104.1 |

**Paired method differences, FRR(A) - FRR(B) on the same 48 wakes per k (cross-speaker, clean, target 1 FA/h; 95 % bootstrap over words; in brackets after the interval: wakes where A has the lower FRR : wakes where B has). 'oracle/wake' = each wake's threshold set to 1 FA/h on its own evaluation negatives, so both methods run at about the same FA/h and the difference is a difference of DET operating points; 'bank tail' = the deployable bank rule, where the two methods may also differ in achieved FA/h (see the table above).**

| method A | method B | oracle/wake k=3 | oracle/wake k=6 | oracle/wake k=10 | bank tail k=3 | bank tail k=6 | bank tail k=10 |
|---|---|---|---|---|---|---|---|
| clean templates only | deployed matcher (noise-augmented templates) | -0.117 [-0.145, -0.092] (45:0) | -0.160 [-0.194, -0.125] (48:0) | -0.167 [-0.206, -0.127] (46:2) | -0.075 [-0.106, -0.045] (36:5) | -0.118 [-0.150, -0.084] (44:3) | -0.126 [-0.161, -0.095] (44:2) |
| deployed matcher (noise-augmented templates) | deployed, no normaliser | -0.015 [-0.036, +0.003] (21:13) | -0.019 [-0.032, -0.007] (27:9) | -0.029 [-0.047, -0.013] (28:8) | -0.016 [-0.040, +0.009] (25:12) | -0.026 [-0.049, -0.009] (24:10) | -0.031 [-0.053, -0.012] (24:8) |
| clean templates, bank normaliser | clean templates only | -0.002 [-0.008, +0.003] (18:12) | -0.008 [-0.015, -0.002] (22:12) | -0.010 [-0.015, -0.004] (21:8) | -0.003 [-0.010, +0.005] (15:14) | -0.000 [-0.009, +0.009] (17:11) | +0.003 [-0.004, +0.011] (10:13) |
| deployed, best single template | deployed matcher (noise-augmented templates) | -0.020 [-0.026, -0.013] (28:5) | -0.012 [-0.022, -0.001] (19:8) | -0.007 [-0.017, +0.002] (20:15) | -0.017 [-0.028, -0.007] (27:7) | -0.008 [-0.018, -0.000] (20:9) | -0.007 [-0.016, +0.004] (25:10) |
| clean templates only | embedding part only (96-d) | -0.070 [-0.092, -0.049] (37:7) | -0.117 [-0.144, -0.090] (42:3) | -0.117 [-0.149, -0.087] (43:4) | -0.053 [-0.078, -0.025] (36:8) | -0.112 [-0.141, -0.085] (45:3) | -0.110 [-0.146, -0.081] (44:2) |
| clean templates only | log-mel part only (32-d) | -0.171 [-0.203, -0.140] (48:0) | -0.213 [-0.256, -0.170] (48:0) | -0.229 [-0.270, -0.193] (48:0) | -0.123 [-0.153, -0.094] (46:0) | -0.169 [-0.203, -0.133] (47:0) | -0.178 [-0.220, -0.143] (48:0) |
| clean templates only | prototype (mean embedding) | -0.143 [-0.174, -0.110] (47:1) | -0.184 [-0.226, -0.145] (47:0) | -0.203 [-0.243, -0.166] (48:0) | -0.102 [-0.133, -0.072] (43:3) | -0.149 [-0.181, -0.115] (45:2) | -0.158 [-0.202, -0.120] (46:2) |
| clean templates only | MFCC + DTW | -0.126 [-0.151, -0.102] (47:1) | -0.138 [-0.175, -0.101] (42:4) | -0.154 [-0.200, -0.109] (43:5) | -0.091 [-0.113, -0.071] (43:2) | -0.119 [-0.153, -0.087] (42:2) | -0.125 [-0.174, -0.084] (42:5) |
| deployed matcher (noise-augmented templates) | MFCC + DTW | -0.009 [-0.029, +0.016] (24:16) | +0.022 [-0.012, +0.063] (16:24) | +0.013 [-0.027, +0.055] (21:23) | -0.016 [-0.038, +0.005] (28:16) | -0.001 [-0.036, +0.034] (21:23) | +0.001 [-0.037, +0.042] (20:22) |

**Effect of the number of enrollment shots, FRR(k = a) - FRR(k = b) on the same 48 wakes (nested enrollment, same positives and negatives; cross-speaker, clean, target 1 FA/h; 95 % bootstrap over words)**

| method | oracle/wake k6-k3 | oracle/wake k10-k6 | oracle/wake k10-k3 | bank tail k6-k3 | bank tail k10-k6 | bank tail k10-k3 |
|---|---|---|---|---|---|---|
| deployed matcher (noise-augmented templates) | +0.001 [-0.011, +0.013] | -0.009 [-0.023, +0.005] | -0.008 [-0.026, +0.010] | -0.003 [-0.022, +0.016] | -0.001 [-0.014, +0.011] | -0.005 [-0.028, +0.017] |
| deployed, best single template | +0.009 [-0.007, +0.026] | -0.004 [-0.018, +0.009] | +0.005 [-0.011, +0.022] | +0.006 [-0.015, +0.026] | +0.000 [-0.011, +0.013] | +0.006 [-0.009, +0.022] |
| clean templates only | -0.041 [-0.067, -0.017] | -0.016 [-0.033, -0.000] | -0.057 [-0.083, -0.036] | -0.046 [-0.085, -0.011] | -0.009 [-0.034, +0.017] | -0.055 [-0.104, -0.016] |
| clean templates, bank normaliser | -0.047 [-0.080, -0.017] | -0.018 [-0.039, +0.002] | -0.065 [-0.093, -0.041] | -0.043 [-0.083, -0.006] | -0.006 [-0.028, +0.018] | -0.049 [-0.090, -0.016] |
| deployed, no normaliser | +0.005 [-0.007, +0.017] | +0.001 [-0.006, +0.009] | +0.006 [-0.006, +0.018] | +0.008 [-0.003, +0.017] | +0.003 [-0.005, +0.012] | +0.011 [-0.003, +0.026] |
| embedding part only (96-d) | +0.006 [-0.010, +0.022] | -0.016 [-0.030, +0.000] | -0.010 [-0.031, +0.012] | +0.013 [-0.003, +0.032] | -0.011 [-0.022, -0.000] | +0.002 [-0.019, +0.024] |
| log-mel part only (32-d) | +0.001 [+0.000, +0.002] | +0.000 [+0.000, +0.000] | +0.001 [+0.000, +0.002] | +0.000 [-0.000, +0.001] | -0.000 [-0.001, +0.000] | -0.000 [-0.001, +0.001] |
| prototype (mean embedding) | +0.000 [-0.009, +0.010] | +0.003 [-0.001, +0.007] | +0.003 [-0.006, +0.014] | +0.000 [-0.008, +0.008] | +0.001 [-0.006, +0.008] | +0.001 [-0.009, +0.010] |
| MFCC + DTW | -0.030 [-0.049, -0.014] | +0.000 [-0.015, +0.014] | -0.030 [-0.049, -0.010] | -0.018 [-0.034, -0.003] | -0.003 [-0.014, +0.006] | -0.022 [-0.042, -0.004] |

## 3. Noisy condition (white + pink noise at 10 dB SNR, appearing after enrollment)

k = 6, target 1 FA/h (thresholds from clean enrollment, room audio and bank):

| method | rule | FRR | FA/h mean | FA/h median |
|---|---|---|---|---|
| deployed matcher (noise-augmented templates) | deployed (20 s room audio) | 0.089 | 2079.5 | 2497.0 |
| deployed matcher (noise-augmented templates) | bank 60 min, tail fit | 0.440 | 699.9 | 191.5 |
| deployed matcher (noise-augmented templates) | bank 60 min + noisy copy, empirical | 0.738 | 124.3 | 47.250 |
| deployed matcher (noise-augmented templates) | oracle per wake (eval negatives) | 0.966 | 0.922 | 1.000 |
| clean templates only | deployed (20 s room audio) | 0.704 | 28.062 | 8.125 |
| clean templates only | bank 60 min, tail fit | 0.994 | 0.010 | 0.000 |
| clean templates only | bank 60 min + noisy copy, empirical | 0.981 | 0.078 | 0.000 |
| clean templates only | oracle per wake (eval negatives) | 0.859 | 0.974 | 1.000 |
| clean templates, bank normaliser | bank 60 min, tail fit | 0.996 | 0.005 | 0.000 |
| clean templates, bank normaliser | bank 60 min + noisy copy, empirical | 0.983 | 0.057 | 0.000 |
| clean templates, bank normaliser | oracle per wake (eval negatives) | 0.848 | 0.937 | 1.000 |
| embedding part only (96-d) | bank 60 min, tail fit | 0.955 | 0.854 | 0.500 |
| embedding part only (96-d) | bank 60 min + noisy copy, empirical | 0.903 | 2.844 | 2.750 |
| embedding part only (96-d) | oracle per wake (eval negatives) | 0.926 | 0.922 | 1.000 |
| prototype (mean embedding) | bank 60 min, tail fit | 0.639 | 36.719 | 24.750 |
| prototype (mean embedding) | bank 60 min + noisy copy, empirical | 0.738 | 10.437 | 9.625 |
| prototype (mean embedding) | oracle per wake (eval negatives) | 0.886 | 0.948 | 1.000 |
| MFCC + DTW | bank 60 min, tail fit | 0.995 | 0.016 | 0.000 |
| MFCC + DTW | bank 60 min + noisy copy, empirical | 0.990 | 0.057 | 0.000 |
| MFCC + DTW | oracle per wake (eval negatives) | 0.919 | 0.911 | 1.000 |

Pure noise (no speech, 10 min, same level), seed 1, k = 6, threshold from the clean bank at 1 FA/h (empirical):

| method | noise_only_fah mean | noise_only_fah median | eval_noisy_fah mean | eval_noisy_fah median | eval_clean_fah mean | eval_clean_fah median |
|---|---|---|---|---|---|---|
| nexa | 780.8 | 6.000 | 768.6 | 192.9 | 2.484 | 2.250 |
| nexa-noaug | 0.000 | 0.000 | 0.031 | 0.000 | 2.656 | 2.000 |

Exploratory: condition-matched normaliser (seed 1, bank 60 min tail fit, target 1 FA/h):

| method | k | condition | FRR | FA/h mean | FA/h median | wakes |
|---|---|---|---|---|---|---|
| clean templates, condition-matched normaliser | 3 | clean | 0.845 | 0.812 | 0.750 | 16 |
| clean templates, condition-matched normaliser | 3 | noisy | 0.991 | 0.062 | 0.000 | 16 |
| clean templates, condition-matched normaliser | 6 | clean | 0.831 | 0.828 | 0.750 | 16 |
| clean templates, condition-matched normaliser | 6 | noisy | 0.994 | 0.047 | 0.000 | 16 |
| clean templates, condition-matched normaliser | 10 | clean | 0.823 | 0.703 | 0.250 | 16 |
| clean templates, condition-matched normaliser | 10 | noisy | 0.993 | 0.031 | 0.000 | 16 |
| noise-augmented templates, condition-matched normaliser | 3 | clean | 0.929 | 1.266 | 1.250 | 16 |
| noise-augmented templates, condition-matched normaliser | 3 | noisy | 0.999 | 0.062 | 0.000 | 16 |
| noise-augmented templates, condition-matched normaliser | 6 | clean | 0.932 | 1.109 | 1.125 | 16 |
| noise-augmented templates, condition-matched normaliser | 6 | noisy | 1.000 | 0.047 | 0.000 | 16 |
| noise-augmented templates, condition-matched normaliser | 10 | clean | 0.910 | 1.141 | 1.000 | 16 |
| noise-augmented templates, condition-matched normaliser | 10 | noisy | 0.999 | 0.062 | 0.000 | 16 |

## 3b. Out-of-domain negatives: continuous read speech (FLEURS Persian test split)

- 871 utterances, 3.76 h with 0.25 s gaps (3.70 h of speech). Utterances containing the target word are masked for that wake: 0-185 per wake, leaving 2.90-3.76 h.
- Same wakes (cross-speaker, k = 6, 48 wakes) and the same thresholds as section 1; FRR is the MSWC positive FRR at that threshold.

Target 1 FA/h (oracle per wake = threshold chosen on FLEURS itself):

| method | rule | FRR (MSWC positives) | FA/h FLEURS mean | FA/h FLEURS median | share of wakes <= 2 FA/h | FA/h MSWC eval (same threshold) |
|---|---|---|---|---|---|---|
| deployed matcher (noise-augmented templates) | deployed (20 s room audio) | 0.586 | 29.545 | 17.568 | 0.125 | 117.9 |
| deployed matcher (noise-augmented templates) | one global threshold (bank, other wakes) | 0.975 | 0.404 | 0.000 | 0.917 | 2.594 |
| deployed matcher (noise-augmented templates) | bank 60 min, empirical | 0.920 | 1.051 | 0.532 | 0.875 | 2.568 |
| deployed matcher (noise-augmented templates) | bank 60 min, order statistic | 0.950 | 0.431 | 0.267 | 0.958 | 1.151 |
| deployed matcher (noise-augmented templates) | bank 60 min, tail fit | 0.947 | 0.466 | 0.266 | 0.938 | 1.151 |
| deployed matcher (noise-augmented templates) | oracle per wake (eval negatives) | 0.874 | 0.749 | 0.807 | 1.000 | 7.625 |
| clean templates only | deployed (20 s room audio) | 0.277 | 9.185 | 5.586 | 0.208 | 106.2 |
| clean templates only | one global threshold (bank, other wakes) | 0.860 | 0.039 | 0.000 | 1.000 | 1.005 |
| clean templates only | bank 60 min, empirical | 0.745 | 0.017 | 0.000 | 1.000 | 2.120 |
| clean templates only | bank 60 min, order statistic | 0.820 | 0.006 | 0.000 | 1.000 | 0.896 |
| clean templates only | bank 60 min, tail fit | 0.828 | 0.006 | 0.000 | 1.000 | 0.792 |
| clean templates only | oracle per wake (eval negatives) | 0.438 | 0.806 | 0.809 | 1.000 | 21.828 |
| clean templates, bank normaliser | one global threshold (bank, other wakes) | 0.864 | 0.028 | 0.000 | 1.000 | 1.000 |
| clean templates, bank normaliser | bank 60 min, empirical | 0.738 | 0.040 | 0.000 | 1.000 | 2.010 |
| clean templates, bank normaliser | bank 60 min, order statistic | 0.815 | 0.011 | 0.000 | 1.000 | 0.937 |
| clean templates, bank normaliser | bank 60 min, tail fit | 0.828 | 0.011 | 0.000 | 1.000 | 0.766 |
| clean templates, bank normaliser | oracle per wake (eval negatives) | 0.423 | 0.777 | 0.808 | 1.000 | 22.266 |
| embedding part only (96-d) | one global threshold (bank, other wakes) | 0.944 | 0.182 | 0.000 | 1.000 | 1.370 |
| embedding part only (96-d) | bank 60 min, empirical | 0.876 | 0.308 | 0.133 | 1.000 | 1.818 |
| embedding part only (96-d) | bank 60 min, order statistic | 0.926 | 0.149 | 0.000 | 1.000 | 0.901 |
| embedding part only (96-d) | bank 60 min, tail fit | 0.941 | 0.099 | 0.000 | 1.000 | 0.552 |
| embedding part only (96-d) | oracle per wake (eval negatives) | 0.744 | 0.783 | 0.807 | 1.000 | 8.005 |
| MFCC + DTW | one global threshold (bank, other wakes) | 0.975 | 2.109 | 0.000 | 0.917 | 1.542 |
| MFCC + DTW | bank 60 min, empirical | 0.920 | 0.548 | 0.000 | 0.958 | 1.417 |
| MFCC + DTW | bank 60 min, order statistic | 0.950 | 0.252 | 0.000 | 0.979 | 0.781 |
| MFCC + DTW | bank 60 min, tail fit | 0.948 | 0.201 | 0.000 | 1.000 | 0.687 |
| MFCC + DTW | oracle per wake (eval negatives) | 0.864 | 0.749 | 0.805 | 1.000 | 6.073 |

## 4. Same-speaker enrollment (MSWC)

- Main set: 66 evaluation speakers with a word repeated >= 15 times (one word per speaker; words: این x28, است x18, آیا x10, برای x3, دارم x2, توانم x2, لطفا x1, کنم x1, خواهم x1); k = 3, 6, 10 nested; 857 positives in total (5-20 per speaker).
- Broad set (k = 3): 134 speakers with >= 8 repeats, 1648 positives (enroll 3, test on up to 20 others); words: این x59, است x33, آیا x11, برای x9, کنم x5, شما x4, دارم x3, توانید x2, توانم x2, لطفا x1, نمی x1, بود x1, هیچ x1, اون x1, خواهم x1.
- What MSWC offers: speaker-word pairs with >= 8: 312 pairs (134 speakers, 25 words); >= 15: 111 pairs (66 speakers, 11 words); >= 20: 62 pairs (40 speakers, 9 words); >= 30: 27 pairs (22 speakers, 4 words). We use one word per speaker (the most repeated), so the speakers are independent. The repeated words are mostly very frequent one- or two-syllable function words (این, است and آیا are 56 of the 66 main-set words) and only partly overlap the cross-speaker targets, so the two settings differ in the words as well as in the speakers. The repeats of a speaker are cut from different Common Voice sentences read by that speaker; enrollment and test may share session and microphone, unlike a real far-field room.
- Every speaker in both sets is labelled MALE in MSWC (all 3,197 test speakers: NAN 1,972, MALE 1,020, FEMALE 200, OTHER 5).
- Own-voice negatives: the same speaker saying other words (outside the bank vocabulary, up to 400 clips per speaker); FA/h pooled over speakers. In the main set 88 of 12609 own-voice clips are words that contain the target as a substring (e.g. اینجا for این); they are kept as negatives.
- Other-speaker negatives: the 4 h seed-1 evaluation stream of section 1.
- A wake needs at least 3 accepted enrollment recordings (as in NexaRoom). Wakes built / planned: main k = 3: 63/66; main k = 6: 66/66; main k = 10: 66/66; broad k = 3: 130/134. Wakes that could not be built are left out of the averages below.

Clean condition, target 1 FA/h (main set at k = 10 and broad set at k = 3; main set at k = 3 and 6 in the k table below and in `results/tables/samespk_rules.csv`). 'oracle per wake' is set on the other-speaker negatives; own-voice FA/h is measured at the same threshold:

| set | k | method | rule | speakers | FRR [95% CI] | FA/h other speakers | FA/h own voice (pooled) | own-voice hours |
|---|---|---|---|---|---|---|---|---|
| main | 10 | deployed matcher (noise-augmented templates) | deployed (20 s room audio) | 66 | 0.521 [0.444, 0.586] | 96.579 | 170.7 | 4.382 |
| main | 10 | deployed matcher (noise-augmented templates) | one global threshold (bank, other wakes) | 66 | 0.827 [0.762, 0.887] | 1.242 | 2.967 | 4.382 |
| main | 10 | deployed matcher (noise-augmented templates) | bank 60 min, tail fit | 66 | 0.797 [0.726, 0.860] | 1.106 | 3.423 | 4.382 |
| main | 10 | deployed matcher (noise-augmented templates) | oracle per wake (eval negatives) | 66 | 0.792 [0.727, 0.854] | 0.932 | 2.510 | 4.382 |
| main | 10 | deployed, best single template | one global threshold (bank, other wakes) | 66 | 0.823 [0.756, 0.880] | 1.314 | 2.738 | 4.382 |
| main | 10 | deployed, best single template | bank 60 min, tail fit | 66 | 0.788 [0.721, 0.854] | 1.133 | 2.738 | 4.382 |
| main | 10 | deployed, best single template | oracle per wake (eval negatives) | 66 | 0.776 [0.704, 0.842] | 0.939 | 2.738 | 4.382 |
| main | 10 | clean templates only | deployed (20 s room audio) | 66 | 0.223 [0.178, 0.278] | 100.5 | 184.8 | 4.382 |
| main | 10 | clean templates only | one global threshold (bank, other wakes) | 66 | 0.650 [0.578, 0.719] | 1.114 | 2.967 | 4.382 |
| main | 10 | clean templates only | bank 60 min, tail fit | 66 | 0.652 [0.587, 0.717] | 0.905 | 5.477 | 4.382 |
| main | 10 | clean templates only | oracle per wake (eval negatives) | 66 | 0.620 [0.550, 0.686] | 0.939 | 5.705 | 4.382 |
| main | 10 | clean templates, bank normaliser | one global threshold (bank, other wakes) | 66 | 0.638 [0.560, 0.711] | 1.117 | 3.651 | 4.382 |
| main | 10 | clean templates, bank normaliser | bank 60 min, tail fit | 66 | 0.632 [0.561, 0.693] | 0.894 | 4.792 | 4.382 |
| main | 10 | clean templates, bank normaliser | oracle per wake (eval negatives) | 66 | 0.596 [0.534, 0.664] | 0.947 | 5.477 | 4.382 |
| main | 10 | embedding part only (96-d) | one global threshold (bank, other wakes) | 66 | 0.830 [0.769, 0.887] | 1.258 | 2.738 | 4.382 |
| main | 10 | embedding part only (96-d) | bank 60 min, tail fit | 66 | 0.808 [0.735, 0.874] | 0.875 | 2.054 | 4.382 |
| main | 10 | embedding part only (96-d) | oracle per wake (eval negatives) | 66 | 0.781 [0.705, 0.847] | 0.943 | 2.510 | 4.382 |
| main | 10 | prototype (mean embedding) | one global threshold (bank, other wakes) | 66 | 0.962 [0.937, 0.982] | 0.860 | 4.792 | 4.382 |
| main | 10 | prototype (mean embedding) | bank 60 min, tail fit | 66 | 0.964 [0.939, 0.983] | 0.583 | 3.423 | 4.382 |
| main | 10 | prototype (mean embedding) | oracle per wake (eval negatives) | 66 | 0.944 [0.919, 0.968] | 0.928 | 4.564 | 4.382 |
| main | 10 | MFCC + DTW | one global threshold (bank, other wakes) | 66 | 0.569 [0.490, 0.641] | 1.307 | 28.526 | 4.382 |
| main | 10 | MFCC + DTW | bank 60 min, tail fit | 66 | 0.448 [0.374, 0.521] | 1.163 | 56.595 | 4.382 |
| main | 10 | MFCC + DTW | oracle per wake (eval negatives) | 66 | 0.423 [0.351, 0.496] | 0.951 | 51.574 | 4.382 |
| broad | 3 | deployed matcher (noise-augmented templates) | deployed (20 s room audio) | 130 | 0.489 [0.442, 0.536] | 110.3 | 183.7 | 6.713 |
| broad | 3 | deployed matcher (noise-augmented templates) | one global threshold (bank, other wakes) | 130 | 0.883 [0.848, 0.915] | 0.940 | 2.830 | 6.713 |
| broad | 3 | deployed matcher (noise-augmented templates) | bank 60 min, tail fit | 130 | 0.851 [0.812, 0.891] | 0.871 | 2.830 | 6.713 |
| broad | 3 | deployed matcher (noise-augmented templates) | oracle per wake (eval negatives) | 130 | 0.838 [0.797, 0.876] | 0.935 | 2.979 | 6.713 |
| broad | 3 | deployed, best single template | one global threshold (bank, other wakes) | 130 | 0.859 [0.819, 0.893] | 1.044 | 1.936 | 6.713 |
| broad | 3 | deployed, best single template | bank 60 min, tail fit | 130 | 0.832 [0.795, 0.869] | 0.958 | 3.128 | 6.713 |
| broad | 3 | deployed, best single template | oracle per wake (eval negatives) | 130 | 0.812 [0.770, 0.851] | 0.925 | 3.277 | 6.713 |
| broad | 3 | clean templates only | deployed (20 s room audio) | 130 | 0.244 [0.206, 0.283] | 103.6 | 163.4 | 6.713 |
| broad | 3 | clean templates only | one global threshold (bank, other wakes) | 130 | 0.732 [0.683, 0.776] | 1.029 | 2.979 | 6.713 |
| broad | 3 | clean templates only | bank 60 min, tail fit | 130 | 0.700 [0.653, 0.744] | 0.852 | 4.767 | 6.713 |
| broad | 3 | clean templates only | oracle per wake (eval negatives) | 130 | 0.663 [0.622, 0.704] | 0.952 | 5.512 | 6.713 |
| broad | 3 | clean templates, bank normaliser | one global threshold (bank, other wakes) | 130 | 0.735 [0.689, 0.781] | 1.060 | 2.830 | 6.713 |
| broad | 3 | clean templates, bank normaliser | bank 60 min, tail fit | 130 | 0.698 [0.654, 0.740] | 0.829 | 5.065 | 6.713 |
| broad | 3 | clean templates, bank normaliser | oracle per wake (eval negatives) | 130 | 0.655 [0.613, 0.700] | 0.954 | 6.256 | 6.713 |
| broad | 3 | embedding part only (96-d) | one global threshold (bank, other wakes) | 130 | 0.888 [0.853, 0.916] | 0.979 | 1.192 | 6.713 |
| broad | 3 | embedding part only (96-d) | bank 60 min, tail fit | 130 | 0.852 [0.810, 0.892] | 0.669 | 2.085 | 6.713 |
| broad | 3 | embedding part only (96-d) | oracle per wake (eval negatives) | 130 | 0.817 [0.777, 0.854] | 0.963 | 2.681 | 6.713 |
| broad | 3 | prototype (mean embedding) | one global threshold (bank, other wakes) | 130 | 0.960 [0.944, 0.975] | 0.958 | 1.788 | 6.713 |
| broad | 3 | prototype (mean embedding) | bank 60 min, tail fit | 130 | 0.944 [0.925, 0.962] | 0.617 | 1.936 | 6.713 |
| broad | 3 | prototype (mean embedding) | oracle per wake (eval negatives) | 130 | 0.928 [0.904, 0.949] | 0.927 | 3.277 | 6.713 |
| broad | 3 | MFCC + DTW | one global threshold (bank, other wakes) | 130 | 0.715 [0.661, 0.764] | 1.131 | 15.790 | 6.713 |
| broad | 3 | MFCC + DTW | bank 60 min, tail fit | 130 | 0.584 [0.529, 0.639] | 0.981 | 32.473 | 6.713 |
| broad | 3 | MFCC + DTW | oracle per wake (eval negatives) | 130 | 0.550 [0.492, 0.604] | 0.952 | 40.070 | 6.713 |

Paired method differences, same speaker, FRR(A) - FRR(B) on the same wakes (clean, target 1 FA/h; 95 % bootstrap over speakers):

| method A | method B | main k=10, oracle/wake | main k=10, bank tail | broad k=3, oracle/wake | broad k=3, bank tail |
|---|---|---|---|---|---|
| clean templates only | deployed matcher (noise-augmented templates) | -0.172 [-0.213, -0.136] (n=66) | -0.145 [-0.185, -0.106] (n=66) | -0.175 [-0.205, -0.147] (n=130) | -0.151 [-0.180, -0.125] (n=130) |
| clean templates, bank normaliser | clean templates only | -0.023 [-0.035, -0.012] (n=66) | -0.020 [-0.034, -0.007] (n=66) | -0.008 [-0.016, +0.000] (n=130) | -0.001 [-0.009, +0.006] (n=130) |
| deployed, best single template | deployed matcher (noise-augmented templates) | -0.016 [-0.041, +0.005] (n=66) | -0.010 [-0.026, +0.006] (n=66) | -0.026 [-0.040, -0.013] (n=130) | -0.019 [-0.033, -0.005] (n=130) |
| clean templates only | embedding part only (96-d) | -0.162 [-0.198, -0.127] (n=66) | -0.156 [-0.193, -0.119] (n=66) | -0.154 [-0.179, -0.128] (n=130) | -0.152 [-0.180, -0.123] (n=130) |
| clean templates only | prototype (mean embedding) | -0.325 [-0.385, -0.266] (n=66) | -0.313 [-0.375, -0.247] (n=66) | -0.264 [-0.298, -0.231] (n=130) | -0.245 [-0.283, -0.207] (n=130) |
| clean templates only | MFCC + DTW | +0.197 [+0.133, +0.259] (n=66) | +0.203 [+0.145, +0.263] (n=66) | +0.114 [+0.073, +0.155] (n=130) | +0.115 [+0.077, +0.155] (n=130) |
| deployed matcher (noise-augmented templates) | MFCC + DTW | +0.369 [+0.313, +0.424] (n=66) | +0.349 [+0.289, +0.411] (n=66) | +0.289 [+0.243, +0.336] (n=130) | +0.266 [+0.223, +0.310] (n=130) |

Noisy condition (main set), target 1 FA/h:

| k | method | rule | FRR | FA/h |
|---|---|---|---|---|
| 3 | deployed matcher (noise-augmented templates) | bank 60 min, tail fit | 0.577 | 225.4 |
| 3 | deployed matcher (noise-augmented templates) | oracle per wake (eval negatives) | 0.821 | 0.913 |
| 3 | clean templates only | bank 60 min, tail fit | 0.984 | 0.012 |
| 3 | clean templates only | oracle per wake (eval negatives) | 0.772 | 0.948 |
| 3 | MFCC + DTW | bank 60 min, tail fit | 0.941 | 0.044 |
| 3 | MFCC + DTW | oracle per wake (eval negatives) | 0.651 | 0.952 |
| 6 | deployed matcher (noise-augmented templates) | bank 60 min, tail fit | 0.493 | 358.6 |
| 6 | deployed matcher (noise-augmented templates) | oracle per wake (eval negatives) | 0.810 | 0.924 |
| 6 | clean templates only | bank 60 min, tail fit | 0.982 | 0.023 |
| 6 | clean templates only | oracle per wake (eval negatives) | 0.761 | 0.936 |
| 6 | MFCC + DTW | bank 60 min, tail fit | 0.933 | 0.038 |
| 6 | MFCC + DTW | oracle per wake (eval negatives) | 0.592 | 0.920 |
| 10 | deployed matcher (noise-augmented templates) | bank 60 min, tail fit | 0.460 | 482.4 |
| 10 | deployed matcher (noise-augmented templates) | oracle per wake (eval negatives) | 0.829 | 0.917 |
| 10 | clean templates only | bank 60 min, tail fit | 0.983 | 0.008 |
| 10 | clean templates only | oracle per wake (eval negatives) | 0.760 | 0.928 |
| 10 | MFCC + DTW | bank 60 min, tail fit | 0.929 | 0.027 |
| 10 | MFCC + DTW | oracle per wake (eval negatives) | 0.559 | 0.939 |

Main set, only speakers whose wake was built at all three k (paired over k), clean, target 1 FA/h:

| method | rule | speakers | k=3 FRR / FA/h other / FA/h own | k=6 FRR / FA/h other / FA/h own | k=10 FRR / FA/h other / FA/h own | dFRR k10-k3 [95% CI] |
|---|---|---|---|---|---|---|
| embedding part only (96-d) | bank 60 min, tail fit | 63 | 0.834 / 0.73 / 2.59 | 0.797 / 0.84 / 1.41 | 0.798 / 0.86 / 2.12 | -0.035 [-0.084, +0.002] |
| embedding part only (96-d) | oracle per wake (eval negatives) | 63 | 0.808 / 0.96 / 2.82 | 0.788 / 0.95 / 1.88 | 0.771 / 0.94 / 2.59 | -0.037 [-0.085, +0.006] |
| MFCC + DTW | bank 60 min, tail fit | 63 | 0.559 / 0.93 / 31.29 | 0.469 / 1.15 / 47.52 | 0.434 / 1.18 / 57.40 | -0.125 [-0.176, -0.081] |
| MFCC + DTW | oracle per wake (eval negatives) | 63 | 0.516 / 0.96 / 39.29 | 0.443 / 0.92 / 43.52 | 0.414 / 0.95 / 50.58 | -0.102 [-0.152, -0.051] |
| deployed matcher (noise-augmented templates) | bank 60 min, tail fit | 63 | 0.832 / 0.88 / 2.82 | 0.817 / 0.97 / 2.59 | 0.788 / 1.10 / 3.53 | -0.045 [-0.082, -0.010] |
| deployed matcher (noise-augmented templates) | deployed (20 s room audio) | 63 | 0.531 / 105.39 / 200.44 | 0.514 / 100.28 / 177.86 | 0.506 / 95.48 / 169.62 | -0.025 [-0.078, +0.026] |
| deployed matcher (noise-augmented templates) | oracle per wake (eval negatives) | 63 | 0.816 / 0.94 / 1.88 | 0.797 / 0.97 / 3.06 | 0.782 / 0.93 / 2.59 | -0.035 [-0.070, -0.003] |
| clean templates only | bank 60 min, tail fit | 63 | 0.657 / 1.04 / 5.65 | 0.629 / 1.04 / 8.00 | 0.641 / 0.87 / 5.65 | -0.016 [-0.055, +0.024] |
| clean templates only | deployed (20 s room audio) | 63 | 0.233 / 90.04 / 148.92 | 0.209 / 121.77 / 211.97 | 0.210 / 98.62 / 180.68 | -0.023 [-0.070, +0.021] |
| clean templates only | oracle per wake (eval negatives) | 63 | 0.626 / 0.94 / 6.12 | 0.616 / 0.96 / 7.29 | 0.605 / 0.94 / 5.88 | -0.022 [-0.066, +0.021] |
| prototype (mean embedding) | bank 60 min, tail fit | 63 | 0.948 / 0.54 / 1.65 | 0.948 / 0.56 / 1.88 | 0.963 / 0.58 / 3.29 | +0.015 [+0.001, +0.033] |
| prototype (mean embedding) | oracle per wake (eval negatives) | 63 | 0.925 / 0.92 / 3.76 | 0.932 / 0.93 / 3.76 | 0.943 / 0.93 / 4.47 | +0.018 [+0.001, +0.036] |

## 5. Non-speech wakes (ESC-50, all 50 classes)

- Enrollment: k of the 8 fold-1 clips of the class (loudest 1.5 s); positives: the 32 clips of the class in folds 2-5; negatives: folds 3-5 of the other classes (1.75 h before masking the wake's own clips) and the 4 h seed-1 speech stream; sound bank: folds 1-2 (1.17 h before masking). 3 seeds x 50 classes per k.

Target 1 FA/h (FA/h measured on sounds + speech together):

| k | method | rule | FRR [95% CI] | FA/h sounds | FA/h speech |
|---|---|---|---|---|---|
| 3 | deployed matcher (noise-augmented templates) | deployed (20 s room audio) | 0.148 [0.100, 0.204] | 1386.3 | 54.456 |
| 3 | deployed matcher (noise-augmented templates) | speech bank, tail fit | 0.304 [0.227, 0.390] | 1097.2 | 2.398 |
| 3 | deployed matcher (noise-augmented templates) | speech + sound bank, empirical | 0.991 [0.986, 0.997] | 1.100 | 0.037 |
| 3 | deployed matcher (noise-augmented templates) | oracle per wake (eval negatives) | 0.987 [0.978, 0.994] | 2.552 | 0.028 |
| 3 | clean templates only | deployed (20 s room audio) | 0.145 [0.097, 0.197] | 873.6 | 34.698 |
| 3 | clean templates only | speech bank, tail fit | 0.298 [0.219, 0.385] | 658.2 | 1.735 |
| 3 | clean templates only | speech + sound bank, empirical | 0.962 [0.944, 0.976] | 2.297 | 0.085 |
| 3 | clean templates only | oracle per wake (eval negatives) | 0.948 [0.926, 0.966] | 2.595 | 0.057 |
| 3 | clean templates, bank normaliser | speech bank, tail fit | 0.261 [0.182, 0.338] | 748.3 | 1.533 |
| 3 | clean templates, bank normaliser | speech + sound bank, empirical | 0.959 [0.940, 0.975] | 2.399 | 0.085 |
| 3 | clean templates, bank normaliser | oracle per wake (eval negatives) | 0.941 [0.918, 0.960] | 2.536 | 0.052 |
| 3 | embedding part only (96-d) | speech bank, tail fit | 0.530 [0.455, 0.606] | 316.0 | 0.937 |
| 3 | embedding part only (96-d) | speech + sound bank, empirical | 0.963 [0.942, 0.979] | 1.796 | 0.187 |
| 3 | embedding part only (96-d) | oracle per wake (eval negatives) | 0.954 [0.930, 0.971] | 2.520 | 0.102 |
| 3 | log-mel part only (32-d) | speech bank, tail fit | 0.710 [0.638, 0.783] | 443.7 | 1.563 |
| 3 | log-mel part only (32-d) | speech + sound bank, empirical | 0.994 [0.984, 1.000] | 1.151 | 0.000 |
| 3 | log-mel part only (32-d) | oracle per wake (eval negatives) | 0.992 [0.979, 1.000] | 1.640 | 0.000 |
| 3 | prototype (mean embedding) | speech bank, tail fit | 0.236 [0.157, 0.326] | 1456.2 | 0.940 |
| 3 | prototype (mean embedding) | speech + sound bank, empirical | 0.949 [0.922, 0.972] | 2.552 | 0.023 |
| 3 | prototype (mean embedding) | oracle per wake (eval negatives) | 0.941 [0.911, 0.965] | 2.743 | 0.013 |
| 3 | MFCC + DTW | speech bank, tail fit | 0.385 [0.307, 0.466] | 413.7 | 1.047 |
| 3 | MFCC + DTW | speech + sound bank, empirical | 0.954 [0.933, 0.972] | 2.246 | 0.023 |
| 3 | MFCC + DTW | oracle per wake (eval negatives) | 0.941 [0.918, 0.961] | 2.775 | 0.007 |
| 6 | deployed matcher (noise-augmented templates) | deployed (20 s room audio) | 0.179 [0.119, 0.248] | 1391.8 | 63.361 |
| 6 | deployed matcher (noise-augmented templates) | speech bank, tail fit | 0.364 [0.273, 0.458] | 1052.0 | 2.263 |
| 6 | deployed matcher (noise-augmented templates) | speech + sound bank, empirical | 0.995 [0.990, 0.999] | 1.233 | 0.037 |
| 6 | deployed matcher (noise-augmented templates) | oracle per wake (eval negatives) | 0.990 [0.981, 0.997] | 2.411 | 0.030 |
| 6 | clean templates only | deployed (20 s room audio) | 0.152 [0.096, 0.211] | 939.0 | 42.405 |
| 6 | clean templates only | speech bank, tail fit | 0.312 [0.222, 0.403] | 663.6 | 2.195 |
| 6 | clean templates only | speech + sound bank, empirical | 0.968 [0.950, 0.984] | 2.043 | 0.125 |
| 6 | clean templates only | oracle per wake (eval negatives) | 0.951 [0.926, 0.971] | 2.610 | 0.035 |
| 6 | clean templates, bank normaliser | speech bank, tail fit | 0.256 [0.174, 0.334] | 793.5 | 1.673 |
| 6 | clean templates, bank normaliser | speech + sound bank, empirical | 0.961 [0.941, 0.978] | 2.285 | 0.138 |
| 6 | clean templates, bank normaliser | oracle per wake (eval negatives) | 0.945 [0.922, 0.964] | 2.536 | 0.037 |
| 6 | embedding part only (96-d) | speech bank, tail fit | 0.574 [0.494, 0.653] | 275.0 | 0.855 |
| 6 | embedding part only (96-d) | speech + sound bank, empirical | 0.963 [0.940, 0.982] | 1.338 | 0.335 |
| 6 | embedding part only (96-d) | oracle per wake (eval negatives) | 0.949 [0.920, 0.972] | 2.477 | 0.142 |
| 6 | log-mel part only (32-d) | speech bank, tail fit | 0.779 [0.715, 0.839] | 383.4 | 1.517 |
| 6 | log-mel part only (32-d) | speech + sound bank, empirical | 0.995 [0.989, 1.000] | 1.057 | 0.000 |
| 6 | log-mel part only (32-d) | oracle per wake (eval negatives) | 0.993 [0.981, 1.000] | 1.550 | 0.000 |
| 6 | prototype (mean embedding) | speech bank, tail fit | 0.231 [0.151, 0.318] | 1519.8 | 0.777 |
| 6 | prototype (mean embedding) | speech + sound bank, empirical | 0.953 [0.924, 0.976] | 2.383 | 0.000 |
| 6 | prototype (mean embedding) | oracle per wake (eval negatives) | 0.944 [0.911, 0.972] | 2.771 | 0.002 |
| 6 | MFCC + DTW | speech bank, tail fit | 0.354 [0.277, 0.435] | 433.6 | 0.988 |
| 6 | MFCC + DTW | speech + sound bank, empirical | 0.941 [0.912, 0.968] | 2.438 | 0.000 |
| 6 | MFCC + DTW | oracle per wake (eval negatives) | 0.931 [0.897, 0.959] | 2.806 | 0.000 |

Per class (clean templates, k = 6, speech + sound bank, 1 FA/h), best and worst 10 (full list in `results/esc50/per_class_noaug_k6.csv`):

| class | FRR | FA/h |
|---|---|---|
| crow | 0.646 | 0.584 |
| brushing_teeth | 0.802 | 1.344 |
| dog | 0.844 | 3.390 |
| siren | 0.885 | 0.351 |
| vacuum_cleaner | 0.896 | 0.175 |
| clock_tick | 0.896 | 1.578 |
| rain | 0.906 | 0.409 |
| clapping | 0.938 | 1.695 |
| clock_alarm | 0.948 | 2.922 |
| can_opening | 0.948 | 0.584 |
| sea_waves | 1.000 | 0.058 |
| rooster | 1.000 | 0.818 |
| laughing | 1.000 | 0.643 |
| sneezing | 1.000 | 2.104 |
| snoring | 1.000 | 0.117 |
| toilet_flush | 1.000 | 0.000 |
| train | 1.000 | 0.000 |
| pig | 1.000 | 0.994 |
| washing_machine | 1.000 | 0.117 |
| water_drops | 1.000 | 0.468 |

Sounds a user could make on purpose (mean over 3 seeds; FA/h on sounds + speech; bank cal. = speech + sound bank, empirical, 1 FA/h):

| class | deployed tmpl, room cal., k=3: FRR / FA/h | deployed tmpl, room cal., k=6: FRR / FA/h | deployed tmpl, bank cal., k=3: FRR / FA/h | deployed tmpl, bank cal., k=6: FRR / FA/h | clean tmpl, room cal., k=3: FRR / FA/h | clean tmpl, room cal., k=6: FRR / FA/h | clean tmpl, bank cal., k=3: FRR / FA/h | clean tmpl, bank cal., k=6: FRR / FA/h |
|---|---|---|---|---|---|---|---|---|
| clapping | 0.12 / 367.9 | 0.10 / 372.4 | 0.96 / 1.2 | 0.97 / 1.6 | 0.12 / 244.8 | 0.16 / 265.9 | 0.91 / 1.6 | 0.94 / 1.7 |
| door_wood_knock | 0.34 / 607.4 | 0.26 / 670.9 | 1.00 / 0.3 | 1.00 / 0.2 | 0.10 / 294.2 | 0.09 / 288.5 | 0.96 / 0.2 | 0.99 / 1.0 |
| mouse_click | 0.03 / 559.3 | 0.09 / 614.7 | 0.99 / 0.2 | 0.99 / 0.2 | 0.11 / 303.4 | 0.01 / 368.1 | 0.96 / 0.4 | 0.99 / 0.6 |
| keyboard_typing | 0.01 / 548.4 | 0.00 / 636.9 | 1.00 / 0.4 | 1.00 / 0.1 | 0.00 / 188.9 | 0.00 / 353.3 | 0.95 / 0.4 | 0.97 / 0.1 |
| can_opening | 0.18 / 579.4 | 0.47 / 652.8 | 1.00 / 0.4 | 1.00 / 0.4 | 0.15 / 113.2 | 0.40 / 92.9 | 0.94 / 0.9 | 0.95 / 0.6 |
| clock_alarm | 0.20 / 624.4 | 0.38 / 542.9 | 1.00 / 0.1 | 1.00 / 0.1 | 0.03 / 469.6 | 0.01 / 597.2 | 0.98 / 2.2 | 0.95 / 2.9 |
| glass_breaking | 0.31 / 473.4 | 0.40 / 449.2 | 1.00 / 0.4 | 1.00 / 0.2 | 0.30 / 230.3 | 0.23 / 387.0 | 0.97 / 0.6 | 0.99 / 0.4 |
| coughing | 0.40 / 428.6 | 0.76 / 301.4 | 1.00 / 0.6 | 1.00 / 0.3 | 0.80 / 35.0 | 0.65 / 72.1 | 0.98 / 0.6 | 1.00 / 0.4 |
| sneezing | 0.80 / 212.7 | 0.79 / 297.7 | 1.00 / 1.1 | 1.00 / 1.0 | 0.51 / 286.9 | 0.65 / 123.1 | 1.00 / 2.1 | 1.00 / 2.1 |
| laughing | 0.07 / 530.9 | 0.03 / 585.9 | 1.00 / 0.8 | 1.00 / 0.3 | 0.18 / 385.7 | 0.20 / 380.6 | 1.00 / 0.5 | 1.00 / 0.6 |

## 5b. Non-speech wakes: does the sound bank need the same sound types? (ESC-50, class-disjoint)

- The 50 classes are split once into two halves of 25 (five per ESC-50 major group, seed 2024). Bank streams (folds 1-2): 0.58 h and 0.58 h; evaluation streams (folds 3-5): 0.88 h and 0.88 h, before masking the wake's own clips.
- *seen*: bank and evaluation sounds from the same half (different recordings); *unseen*: bank from one half, evaluation sounds from the other. Both use the 60-min speech bank and the 4 h speech stream as well, and the same wakes, templates and positives as section 5. Empirical rule; each condition is the mean of its two half pairings.

Target 1 FA/h (FA/h on speech + sound negatives together unless stated):

| k | method | bank sounds | wakes | FRR [95% CI] | FA/h mean | FA/h median | FA/h sounds only | wakes <= 2 FA/h |
|---|---|---|---|---|---|---|---|---|
| 6 | clean templates only | seen | 150 | 0.958 [0.939, 0.975] | 0.587 | 0.411 | 2.799 | 0.953 |
| 6 | clean templates only | unseen | 150 | 0.958 [0.938, 0.974] | 0.676 | 0.414 | 3.318 | 0.933 |
| 6 | deployed matcher (noise-augmented templates) | seen | 150 | 0.991 [0.984, 0.996] | 0.390 | 0.308 | 2.066 | 1.000 |
| 6 | deployed matcher (noise-augmented templates) | unseen | 150 | 0.991 [0.984, 0.997] | 0.488 | 0.310 | 2.622 | 0.967 |
| 3 | clean templates only | seen | 150 | 0.952 [0.929, 0.969] | 0.582 | 0.513 | 2.850 | 0.987 |
| 3 | clean templates only | unseen | 150 | 0.952 [0.930, 0.969] | 0.630 | 0.515 | 3.124 | 0.973 |
| 3 | deployed matcher (noise-augmented templates) | seen | 150 | 0.988 [0.981, 0.995] | 0.387 | 0.309 | 2.058 | 1.000 |
| 3 | deployed matcher (noise-augmented templates) | unseen | 150 | 0.988 [0.981, 0.994] | 0.431 | 0.311 | 2.303 | 0.993 |

Paired difference unseen minus seen (same wakes, cluster bootstrap over classes), target 1 FA/h. FRR is identical in the two conditions by construction: both use the thresholds of the same two banks, only the evaluation half is swapped.

| k | method | wakes | dFA/h [95% CI] | dFA/h sounds only [95% CI] |
|---|---|---|---|---|
| 6 | clean templates only | 150 | +0.09 [+0.02, +0.17] | +0.52 [+0.13, +0.97] |
| 6 | deployed matcher (noise-augmented templates) | 150 | +0.10 [+0.02, +0.19] | +0.56 [+0.13, +1.03] |
| 3 | clean templates only | 150 | +0.05 [+0.00, +0.10] | +0.27 [+0.00, +0.57] |
| 3 | deployed matcher (noise-augmented templates) | 150 | +0.04 [-0.02, +0.11] | +0.24 [-0.07, +0.59] |

## 5c. Non-speech wakes, same-source enrollment (ESC-50 takes of one recording)

- 38 recordings with >= 4 takes, from 22 classes; k = 3 takes enrolled, the other takes of the same recording are the positives; up to 3 distinct enrollment subsets per recording: 114 wakes, 204 positive trials in total (1-5 per wake). Negatives and banks as in section 5; 95% intervals: cluster bootstrap over recordings.
- The cross-recording rows use the main k = 3 run restricted to the same classes (other recordings of the class as positives).

k = 3, target 1 FA/h (FA/h on sounds + speech):

| method | rule | wakes | same source: FRR [95% CI] | same source: FA/h mean | same source: FA/h median | cross-recording, same classes: FRR | cross-recording: FA/h mean |
|---|---|---|---|---|---|---|---|
| deployed matcher (noise-augmented templates) | deployed (20 s room audio) | 114 | 0.046 [0.012, 0.085] | 407.0 | 399.2 | 0.104 | 459.5 |
| deployed matcher (noise-augmented templates) | speech bank, tail fit | 114 | 0.103 [0.038, 0.190] | 344.1 | 365.7 | 0.210 | 350.1 |
| deployed matcher (noise-augmented templates) | speech + sound bank, empirical | 114 | 0.937 [0.873, 0.987] | 0.360 | 0.175 | 0.994 | 0.388 |
| deployed matcher (noise-augmented templates) | speech + sound bank, tail fit | 114 | 0.937 [0.865, 0.987] | 0.201 | 0.000 | 0.997 | 0.191 |
| deployed matcher (noise-augmented templates) | oracle per wake (eval negatives) | 114 | 0.934 [0.860, 0.987] | 0.761 | 0.877 | 0.990 | 0.778 |
| clean templates only | deployed (20 s room audio) | 114 | 0.050 [0.011, 0.098] | 242.2 | 243.0 | 0.113 | 304.1 |
| clean templates only | speech bank, tail fit | 114 | 0.078 [0.021, 0.158] | 217.4 | 239.3 | 0.239 | 217.5 |
| clean templates only | speech + sound bank, empirical | 114 | 0.769 [0.662, 0.865] | 0.818 | 0.526 | 0.959 | 0.890 |
| clean templates only | speech + sound bank, tail fit | 114 | 0.804 [0.693, 0.901] | 0.520 | 0.351 | 0.972 | 0.579 |
| clean templates only | oracle per wake (eval negatives) | 114 | 0.743 [0.620, 0.851] | 0.809 | 0.877 | 0.954 | 0.816 |
| clean templates, bank normaliser | speech bank, tail fit | 114 | 0.059 [0.009, 0.127] | 243.8 | 273.4 | 0.196 | 253.3 |
| clean templates, bank normaliser | speech + sound bank, empirical | 114 | 0.772 [0.667, 0.868] | 0.763 | 0.614 | 0.955 | 0.866 |
| clean templates, bank normaliser | speech + sound bank, tail fit | 114 | 0.825 [0.719, 0.912] | 0.495 | 0.351 | 0.964 | 0.563 |
| clean templates, bank normaliser | oracle per wake (eval negatives) | 114 | 0.759 [0.645, 0.864] | 0.807 | 0.877 | 0.947 | 0.784 |
| embedding part only (96-d) | speech bank, tail fit | 114 | 0.347 [0.247, 0.449] | 94.910 | 96.961 | 0.493 | 111.1 |
| embedding part only (96-d) | speech + sound bank, empirical | 114 | 0.875 [0.794, 0.947] | 0.488 | 0.351 | 0.959 | 0.672 |
| embedding part only (96-d) | speech + sound bank, tail fit | 114 | 0.879 [0.803, 0.947] | 0.271 | 0.175 | 0.965 | 0.428 |
| embedding part only (96-d) | oracle per wake (eval negatives) | 114 | 0.807 [0.715, 0.890] | 0.811 | 0.877 | 0.952 | 0.829 |
| prototype (mean embedding) | speech bank, tail fit | 114 | 0.096 [0.035, 0.171] | 453.0 | 472.4 | 0.131 | 463.8 |
| prototype (mean embedding) | speech + sound bank, empirical | 114 | 0.798 [0.689, 0.899] | 0.977 | 0.877 | 0.938 | 0.837 |
| prototype (mean embedding) | speech + sound bank, tail fit | 114 | 0.864 [0.772, 0.952] | 0.548 | 0.351 | 0.949 | 0.433 |
| prototype (mean embedding) | oracle per wake (eval negatives) | 114 | 0.806 [0.706, 0.892] | 0.846 | 0.877 | 0.926 | 0.839 |
| MFCC + DTW | speech bank, tail fit | 114 | 0.034 [0.011, 0.062] | 113.3 | 104.8 | 0.364 | 116.9 |
| MFCC + DTW | speech + sound bank, empirical | 114 | 0.545 [0.414, 0.670] | 0.786 | 0.701 | 0.949 | 0.664 |
| MFCC + DTW | speech + sound bank, tail fit | 114 | 0.604 [0.476, 0.734] | 0.503 | 0.351 | 0.960 | 0.420 |
| MFCC + DTW | oracle per wake (eval negatives) | 114 | 0.520 [0.395, 0.643] | 0.831 | 0.877 | 0.930 | 0.842 |

Per class, clean templates, speech + sound bank (empirical), 1 FA/h (`results/esc50_samesrc/per_class_noaug.csv`):

| class | wakes | FRR | FA/h |
|---|---|---|---|
| footsteps | 3 | 0.000 | 0.292 |
| siren | 3 | 0.000 | 0.117 |
| frog | 9 | 0.333 | 0.994 |
| sheep | 9 | 0.556 | 1.247 |
| door_wood_creaks | 6 | 0.667 | 0.760 |
| wind | 3 | 0.667 | 0.175 |
| clapping | 3 | 0.667 | 1.695 |
| crow | 3 | 0.667 | 2.396 |
| pig | 9 | 0.778 | 1.461 |
| fireworks | 15 | 0.867 | 0.397 |
| helicopter | 12 | 0.889 | 0.468 |
| crying_baby | 6 | 1.000 | 0.584 |
| airplane | 3 | 1.000 | 1.110 |
| cow | 3 | 1.000 | 0.117 |
| brushing_teeth | 3 | 1.000 | 1.403 |
| chirping_birds | 3 | 1.000 | 0.351 |
| insects | 3 | 1.000 | 0.994 |
| hand_saw | 3 | 1.000 | 0.292 |
| drinking_sipping | 3 | 1.000 | 0.760 |
| pouring_water | 3 | 1.000 | 0.993 |
| thunderstorm | 3 | 1.000 | 2.046 |
| washing_machine | 6 | 1.000 | 0.351 |

## 6. Compute cost (single thread, x86 virtual machine; Raspberry Pi not measured)

- Machine: 13th Gen Intel(R) Core(TM) i9-13900H host, VMware virtual machine with 20 vCPUs; the script was pinned to vCPU [19] and used one thread (ONNX Runtime intra/inter-op threads = 1, BLAS and Numba threads = 1). Other jobs were running (load average 10.0 before, 10.4 after, on 20 vCPUs), so the times are upper bounds for this CPU. Clock: thread CPU time; every measurement repeated 3 times, the tables give the median over repeats and, in brackets, the smallest and largest repeat.
- Software: Python 3.12.16, NumPy 2.5.4, ONNX Runtime 1.31.1, Numba 0.68.1.
- Feature extraction per 80 ms frame (deployed `FeatureStream.push`, 2000 frames per repeat): median 3.059 ms [2.820, 3.237], p95 5.215 ms, wall-clock median 3.063 ms; of this the log-mel model takes 0.358 ms and the embedding model 2.527 ms.
- Template length in real enrollments (seed 1, k = 10, 16 words): median 5 frames of 80 ms, p90 7, max 9 (159 templates; the deployed cap is 30). MFCC + DTW templates: median 34 frames of 10 ms (p90 44). Raw lengths: `results/cost/template_lengths.csv`.
- Templates per wake in practice: the deployed enrollment keeps 3 templates per recording (clean + two noise copies), i.e. 9, 18 and 30 for k = 3, 6 and 10; clean templates only: k. Several wakes on one device add their templates.

Per-frame cost of matching (random templates of the given length; the cost does not depend on the values). 'Deployed' = `WakeDetector.process` (NumPy `TemplateMatcher.step`, `wake_scores` and the decision layer) as it runs on the device; 'compiled' = the same recurrence in Numba (`nexawake/fast.py`), amortised over 1,000 frames:

The deployed NumPy matcher loops in Python over the template columns, so its time depends mainly on the longest template: with 5-frame templates it grows from 0.23 ms (3 templates) to 0.38 ms (240 templates), while at 18 templates 30-frame templates take 1.08 ms. The compiled matcher grows roughly in proportion to templates x length (0.0083 ms for 18 x 5, 1.047 ms for 240 x 30).

| templates | frames per template | deployed ms/frame, median [range] | deployed p95 ms | compiled ms/frame [range] | features + deployed, ms/frame | share of real time (features + deployed) | features + compiled, ms/frame |
|---|---|---|---|---|---|---|---|
| 3 | 5 | 0.235 [0.227, 0.239] | 0.465 | 0.0026 [0.0022, 0.0027] | 3.293 | 4.12 % | 3.061 |
| 6 | 5 | 0.227 [0.215, 0.240] | 0.474 | 0.0068 [0.0036, 0.0068] | 3.286 | 4.11 % | 3.066 |
| 10 | 5 | 0.236 [0.213, 0.327] | 0.516 | 0.0061 [0.0045, 0.0103] | 3.295 | 4.12 % | 3.065 |
| 18 | 5 | 0.242 [0.226, 0.248] | 0.549 | 0.0083 [0.0073, 0.0103] | 3.300 | 4.13 % | 3.067 |
| 30 | 5 | 0.240 [0.209, 0.250] | 0.503 | 0.0166 [0.0129, 0.0275] | 3.299 | 4.12 % | 3.075 |
| 60 | 5 | 0.257 [0.252, 0.266] | 0.421 | 0.0327 [0.0310, 0.0360] | 3.315 | 4.14 % | 3.091 |
| 120 | 5 | 0.286 [0.268, 0.290] | 0.596 | 0.0683 [0.0505, 0.0726] | 3.345 | 4.18 % | 3.127 |
| 240 | 5 | 0.376 [0.348, 0.406] | 0.845 | 0.1587 [0.1268, 0.2098] | 3.435 | 4.29 % | 3.217 |
| 3 | 30 | 1.035 [0.656, 1.076] | 2.059 | 0.0090 [0.0071, 0.0093] | 4.094 | 5.12 % | 3.068 |
| 6 | 30 | 0.669 [0.593, 0.976] | 2.163 | 0.0142 [0.0134, 0.0218] | 3.728 | 4.66 % | 3.073 |
| 10 | 30 | 0.851 [0.584, 1.028] | 2.170 | 0.0418 [0.0258, 0.0429] | 3.910 | 4.89 % | 3.101 |
| 18 | 30 | 1.082 [1.081, 1.107] | 2.269 | 0.0529 [0.0403, 0.0571] | 4.141 | 5.18 % | 3.112 |
| 30 | 30 | 1.142 [1.095, 1.151] | 2.292 | 0.0914 [0.0826, 0.1270] | 4.201 | 5.25 % | 3.150 |
| 60 | 30 | 1.147 [0.926, 1.191] | 2.593 | 0.2083 [0.2078, 0.2192] | 4.206 | 5.26 % | 3.267 |
| 120 | 30 | 1.429 [0.797, 1.446] | 3.025 | 0.4556 [0.4083, 0.4923] | 4.488 | 5.61 % | 3.514 |
| 240 | 30 | 2.037 [1.887, 2.191] | 4.164 | 1.0473 [1.0278, 1.1919] | 5.095 | 6.37 % | 4.106 |

MFCC + DTW baseline per 80 ms of audio (8 MFCC frames; MFCC with librosa amortised over 60 s blocks: 0.066 ms [0.041, 0.068]; DTW compiled), next to the embedding matcher with the same number of templates:

| templates | frames per template (10 ms) | compiled DTW ms per 80 ms [range] | MFCC + DTW total, ms per 80 ms | features + compiled matcher (5-frame templates), ms |
|---|---|---|---|---|
| 3 | 34 | 0.037 [0.036, 0.040] | 0.103 | 3.061 |
| 6 | 34 | 0.077 [0.065, 0.080] | 0.144 | 3.066 |
| 10 | 34 | 0.105 [0.087, 0.124] | 0.171 | 3.065 |

Bank calibration for one wake: scoring the 60-min bank (45,002 frames) and counting false-alarm events at all 901 grid thresholds (templates of 5 frames; deployed NumPy matcher measured once):

| templates | compiled matcher, s [range] | event counting, s | deployed NumPy matcher, s |
|---|---|---|---|
| 3 | 0.12 [0.09, 0.13] | 0.05 [0.04, 0.06] | - |
| 6 | 0.20 [0.17, 0.21] | 0.05 [0.05, 0.06] | 10.6 |
| 10 | 0.28 [0.26, 0.30] | 0.05 [0.04, 0.06] | - |
| 18 | 0.42 [0.34, 0.46] | 0.06 [0.05, 0.07] | 11.5 |
| 30 | 0.78 [0.74, 0.79] | 0.04 [0.03, 0.04] | - |

Raw per-frame times: `results/cost/raw_times.npz`; per-repeat rows: `results/cost/raw_*.csv`; machine and software: `results/cost/cost_info.json`.

## 7. Tables and figures for the manuscript

Every table above is generated by `experiments/make_report.py` from the CSV files below; nothing is typed in by hand. Run `experiments/run_analysis.sh` to rebuild all of them from `results/*/tasks/`.

| file (under results/) | content |
|---|---|
| tables/negatives.csv | length of every negative stream and the FA/h resolution |
| tables/mswc_rules.csv | cross-speaker: method x k x condition x target x threshold rule (sections 1-3) |
| tables/mswc_shots.csv | cross-speaker: FRR and FA/h by method and k at the bank tail-fit threshold (section 2) |
| mswc/paired.csv, mswc/paired_methods.csv, mswc/paired_k.csv | paired differences with bootstrap intervals |
| mswc/noise_only.csv | pure-noise diagnostic (section 3) |
| tables/condnorm_rules.csv | condition-matched normaliser, exploratory (section 3) |
| tables/fleurs_rules.csv | FLEURS read-speech negatives (section 3b) |
| tables/samespk_rules.csv, tables/samespk_enrollment.csv, tables/samespk_paired_k.csv | same-speaker enrollment (section 4) |
| tables/esc50_rules.csv, esc50/per_class_noaug_k6.csv | ESC-50 sound wakes (section 5) |
| tables/esc50_xclass.csv | class-disjoint sound bank (section 5b) |
| tables/esc50_samesrc_rules.csv, esc50_samesrc/per_class_noaug.csv | same-source sound wakes (section 5c) |
| tables/cost_matcher.csv, tables/cost_mfcc_dtw.csv, tables/cost_bank_calibration.csv, cost/cost_info.json | compute cost (section 6) |

Figures (`results/figures/`, PNG at 200 dpi and PDF, made by `experiments/figures.py` after this report, which writes the per-class CSVs they read):

| figure | data (under results/) | content |
|---|---|---|
| fig_det_k6 | mswc/det.csv, mswc/calibration.csv | pooled DET curves, cross-speaker k = 6, clean and 10 dB noise; markers: deployed and bank calibration |
| fig_calibration | mswc/per_task.csv | achieved FA/h per wake for every threshold rule, clean templates, target 1 FA/h, k = 3, 6, 10 pooled |
| fig_shots | mswc/calibration.csv, samespk/calibration.csv | FRR vs k at the bank tail-fit threshold, cross-speaker and same speaker (own-voice FA/h printed) |
| fig_esc50_classes | esc50/per_class_noaug_k6.csv | detection rate per ESC-50 class, k = 6 |
| fig_cost | cost/matcher_cost.csv, cost/cost_info.json | per-frame matching cost vs templates |

