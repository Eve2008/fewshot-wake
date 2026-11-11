"""Fixed partition of the Persian MSWC test split and the shipped negative feature bank.

Partition (seed 0, decided once):
  * evaluation words  = the 16 cross-speaker targets + every word that some speaker
    repeats >= 8 times (the same-speaker targets). They never appear in the bank
    or in any negative stream.
  * speakers: every speaker with a >= 8-repeat word is kept for evaluation; of the
    rest, a random set holding ~20 % of all clips becomes bank-only.
  * other words: a random set holding ~30 % of their clips becomes bank-only.
  * bank pool = bank speakers x bank words; evaluation negatives = evaluation
    speakers x evaluation (non-target) words. Bank and evaluation material share
    no speaker and no word.

The bank is 60 min of these clips (1 s words with 0.25 s gaps), stored as fp16
features (and a noisy copy with babble + 1/f noise at 5-20 dB SNR, a noise family
different from the evaluation noise). Per seed, 4 h of evaluation negatives and
20 s of 'room' audio are drawn without overlap between seeds.

Usage: python experiments/build_bank.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402
from common import SR  # noqa: E402

SAME_MIN_REPEATS = 8
BANK_SPEAKER_SHARE = 0.20
BANK_WORD_SHARE = 0.30
BANK_CLIPS = 2880            # 2880 x 1.25 s = 60 min
EVAL_CLIPS = 11520           # 11520 x 1.25 s = 4 h per seed
ROOM_CLIPS = 20
SEEDS = (1, 2, 3)


def partition(table: pd.DataFrame) -> dict:
    pairs = table.groupby(["SPEAKER", "WORD"]).size().reset_index(name="n")
    heavy = pairs[pairs.n >= SAME_MIN_REPEATS]
    same_words = sorted(set(heavy.WORD))
    target_set = sorted(set(common.TARGETS) | set(same_words))
    protected = set(heavy.SPEAKER)

    rng = np.random.default_rng(0)
    counts = table.SPEAKER.value_counts()
    others = [s for s in counts.index if s not in protected]
    rng.shuffle(others)
    bank_speakers, mass = [], 0
    for speaker in others:
        if mass >= BANK_SPEAKER_SHARE * len(table):
            break
        bank_speakers.append(speaker)
        mass += int(counts[speaker])

    nontarget = table[~table.WORD.isin(target_set)]
    wcounts = nontarget.WORD.value_counts()
    words = list(wcounts.index)
    rng.shuffle(words)
    bank_words, mass = [], 0
    for word in words:
        if mass >= BANK_WORD_SHARE * len(nontarget):
            break
        bank_words.append(word)
        mass += int(wcounts[word])
    return {"same_words": same_words, "target_set": target_set, "bank_speakers": sorted(bank_speakers),
            "bank_words": sorted(bank_words)}


def one_over_f(size: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(rng.normal(0, 1, size))
    freqs = np.fft.rfftfreq(size)
    spectrum[1:] /= np.sqrt(freqs[1:])
    spectrum[0] = 0
    noise = np.fft.irfft(spectrum, n=size)
    return noise / max(noise.std(), 1e-9)


def bank_noise(clean: np.ndarray, seed: int, segment_seconds: float = 10.0) -> np.ndarray:
    """Babble (4 shifted copies of the bank) + 1/f noise; SNR uniform in 5-20 dB per 10 s."""

    rng = np.random.default_rng(seed)
    level = common.signal_level(clean)
    babble = np.zeros(clean.size, dtype=np.float64)
    for _ in range(4):
        babble += np.roll(clean, int(rng.integers(60 * SR, clean.size - 60 * SR)))
    babble /= babble.std()
    out = clean.astype(np.float32).copy()
    seg = int(segment_seconds * SR)
    for start in range(0, clean.size, seg):
        end = min(clean.size, start + seg)
        noise = 0.6 * babble[start:end] + 0.4 * one_over_f(end - start, rng)
        noise /= max(noise.std(), 1e-9)
        snr = rng.uniform(5.0, 20.0)
        out[start:end] += (noise * level / (10 ** (snr / 20))).astype(np.float32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    table = pd.read_csv(common.MSWC / "test.csv")
    split = partition(table)
    bank_spk, bank_w = set(split["bank_speakers"]), set(split["bank_words"])
    targets = set(split["target_set"])
    is_target = table.WORD.isin(targets)
    bank_pool = table[table.SPEAKER.isin(bank_spk) & table.WORD.isin(bank_w)]
    eval_pool = table[~table.SPEAKER.isin(bank_spk) & ~table.WORD.isin(bank_w) & ~is_target]

    bank = bank_pool.sample(n=BANK_CLIPS, random_state=0)
    shuffled = eval_pool.sample(frac=1.0, random_state=1)
    per_seed = {}
    cursor = 0
    for seed in SEEDS:
        room = shuffled.iloc[cursor: cursor + ROOM_CLIPS]
        cursor += ROOM_CLIPS
        neg = shuffled.iloc[cursor: cursor + EVAL_CLIPS]
        cursor += EVAL_CLIPS
        per_seed[str(seed)] = {"room": room.LINK.tolist(), "negatives": neg.LINK.tolist(),
                               "negative_words": neg.WORD.tolist(), "negative_speakers": neg.SPEAKER.tolist()}
    eval_speakers = set(table.SPEAKER) - bank_spk
    target_speakers = {w: int(table[(table.WORD == w) & table.SPEAKER.isin(eval_speakers)].SPEAKER.nunique())
                       for w in common.TARGETS}
    stats = {
        "clips_total": len(table), "speakers_total": int(table.SPEAKER.nunique()), "words_total": int(table.WORD.nunique()),
        "same_words": len(split["same_words"]), "target_set": len(targets),
        "bank_speakers": len(bank_spk), "bank_words": len(bank_w),
        "bank_pool_clips": len(bank_pool), "eval_pool_clips": len(eval_pool),
        "bank_clips": BANK_CLIPS, "bank_unique_words": int(bank.WORD.nunique()),
        "bank_unique_speakers": int(bank.SPEAKER.nunique()),
        "eval_clips_per_seed": EVAL_CLIPS, "eval_pool_used": cursor,
        "eval_unique_words_per_seed": {s: len(set(v["negative_words"])) for s, v in per_seed.items()},
        "eval_unique_speakers_per_seed": {s: len(set(v["negative_speakers"])) for s, v in per_seed.items()},
        "target_eval_speakers": target_speakers,
    }
    # Disjointness checks.
    used = [link for v in per_seed.values() for link in v["room"] + v["negatives"]]
    assert len(used) == len(set(used))
    assert not (set(bank.SPEAKER) & set(table[table.LINK.isin(used)].SPEAKER))
    assert not (set(bank.WORD) & set(table[table.LINK.isin(used)].WORD))
    assert not (set(bank.WORD) & targets)
    assert min(target_speakers.values()) >= 70, target_speakers
    out = common.RESULTS / "splits.json"
    common.dump_json(out, {**split, "bank": bank.LINK.tolist(), "seeds": per_seed, "stats": stats})
    print(json.dumps(stats, ensure_ascii=False, indent=1))

    bank_dir = common.RESULTS / "bank"
    bank_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    values, spans = common.stream([common.read(link) for link in bank.LINK], np.random.default_rng(10), common.GAP_SECONDS)
    noisy = bank_noise(values, seed=11)
    info = {"clips": BANK_CLIPS, "seconds": values.size / SR}
    for name, audio in (("clean", values), ("noisy", noisy)):
        pcm = common.to_pcm16(audio)
        emb, ends = common.features(pcm, jobs=args.jobs, scratch=bank_dir)
        mf = common.mfcc_long(pcm, jobs=args.jobs, scratch=bank_dir)
        np.save(bank_dir / f"{name}_feat.npy", emb.astype(np.float16))
        np.save(bank_dir / f"{name}_mfcc.npy", mf.astype(np.float16))
        if name == "clean":
            np.save(bank_dir / "ends.npy", ends)
        info[f"{name}_frames"] = int(emb.shape[0])
        info[f"{name}_feat_MB_fp16"] = round(emb.shape[0] * emb.shape[1] * 2 / 1e6, 2)
        info[f"{name}_mfcc_MB_fp16"] = round(mf.shape[0] * mf.shape[1] * 2 / 1e6, 2)
    np.save(bank_dir / "spans.npy", np.asarray(spans, dtype=np.int64))
    info["seconds_build"] = round(time.time() - started, 1)
    common.dump_json(bank_dir / "bank_info.json", info)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
