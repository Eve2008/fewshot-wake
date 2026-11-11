# 60-minute negative bank (shipped features)

Features of the 60-minute negative bank used for false-alarm calibration, as built by
`experiments/build_bank.py` (see `bank_info.json` for clip count, duration and sizes).

| File | Content |
|---|---|
| `clean_feat.npy`, `noisy_feat.npy` | openWakeWord embedding frames, float16 |
| `clean_mfcc.npy`, `noisy_mfcc.npy` | MFCC frames, float16 |
| `ends.npy`, `spans.npy` | clip boundaries |

The bank is built from Multilingual Spoken Words Corpus (MSWC) clips. These derived features
are redistributed under the MSWC licence, CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/), with attribution to the MSWC authors
(Mazumder et al., 2021). The repository's MIT licence covers the code only.
