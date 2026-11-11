"""Non-speech wakes with a class-disjoint sound bank (ESC-50).

In run_esc50.py the sound bank (folds 1-2) and the sound evaluation negatives
(folds 3-5) contain the same 49 classes, so a bank-calibrated threshold has seen
every kind of negative sound it is tested on. Here the 50 classes are split once
into two halves H1 and H2 of 25 classes, five from each ESC-50 major group
(animals, natural soundscapes, human non-speech, interior/domestic,
exterior/urban), with a fixed random seed. Four sound streams are built: bank and
evaluation (folds 1-2 and 3-5) for each half.

For a wake of class c (in half Hc; Ho is the other half):
  seen    calibrate on speech bank + bank of Ho, evaluate on speech eval + eval of Ho
          (and the same with Hc minus c),
  unseen  calibrate on speech bank + bank of Hc minus c, evaluate on speech eval + eval of Ho
          (and calibrate on Ho, evaluate on Hc minus c).
Averaged over its two pairings, each condition uses the same two thresholds (one per
bank half) and the same two evaluation halves; only which threshold meets which half
differs. FRR is therefore identical by construction, and the FA/h difference measures
what a sound bank loses on sound types it does not contain. The clips of c are masked
out of the streams of its own half.

Templates are rebuilt exactly as in run_esc50.py (same clips and random seeds;
the deployed threshold is checked against the main-run task file); positives
scores and speech event counts are taken from that file.

Usage: python experiments/run_esc50_xclass.py [--jobs 2] [--shots 6,3]
       python experiments/run_esc50_xclass.py --analyse
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402
import run_esc50  # noqa: E402
import run_mswc  # noqa: E402

OUT = common.RESULTS / "esc50_xclass"
MAIN = common.RESULTS / "esc50" / "tasks"
METHODS = ("nexa-noaug", "nexa")
SPLIT_SEED = 2024
TARGETS_FAH = (0.5, 1.0, 2.0, 5.0, 10.0)
BOOT = 2000


def halves(meta: pd.DataFrame) -> dict[str, list[str]]:
    classes = meta.drop_duplicates("category").sort_values("target")
    rng = np.random.default_rng(SPLIT_SEED)
    h1 = []
    for group in range(5):
        names = classes[classes.target // 10 == group].category.tolist()
        h1 += sorted(rng.permutation(names)[:5].tolist())
    h2 = sorted(set(classes.category) - set(h1))
    return {"H1": sorted(h1), "H2": h2}


def prepare(jobs: int) -> None:
    folder = OUT / "streams"
    if (folder / "info.json").is_file():
        return
    folder.mkdir(parents=True, exist_ok=True)
    meta, clips = run_esc50.load_clips()
    split = halves(meta)
    info = {"split_seed": SPLIT_SEED, "halves": split}
    for half, names in split.items():
        for name, folds in (("eval", (3, 4, 5)), ("bank", (1, 2))):
            rows = meta[meta.category.isin(names) & meta.fold.isin(folds)].sample(frac=1.0, random_state=7)
            values, spans = common.stream([np.asarray(clips[i]) for i in rows.index], np.random.default_rng(77),
                                          common.GAP_SECONDS)
            pcm = common.to_pcm16(values)
            emb, ends = common.features(pcm, jobs=jobs, scratch=folder)
            np.save(folder / f"{half}_{name}_feat.npy", emb)
            np.save(folder / f"{half}_{name}_ends.npy", ends)
            common.dump_json(folder / f"{half}_{name}_index.json",
                             {"rows": rows.index.tolist(), "category": rows.category.tolist(), "spans": spans})
            info[f"{half}_{name}"] = {"clips": len(rows), "seconds": pcm.size / common.SR, "frames": int(emb.shape[0])}
            print(half, name, info[f"{half}_{name}"], flush=True)
    common.dump_json(folder / "info.json", info)


def run_task(payload: tuple) -> dict:
    category, seed, shots = payload
    from methods import NegStream, enroll_clips, mask_frames
    from nexawake import fast
    from nexawake.enroll import build_wake

    started = time.time()
    meta, clips = run_esc50.load_clips()
    folder = OUT / "streams"
    info = common.load_json(folder / "info.json")
    own_half = "H1" if category in info["halves"]["H1"] else "H2"
    main = common.load_json(MAIN / f"{category}_seed{seed}_k{shots}.json")
    own = meta[meta.category == category]
    fold1 = own[own.fold == 1].index.tolist()
    rng = np.random.default_rng(seed * 1000 + int(own.target.iloc[0]))
    enroll_idx = list(rng.permutation(fold1)[:shots])
    assert meta.filename[enroll_idx].tolist() == main["enroll_files"]
    enroll = [run_esc50.loudest(np.asarray(clips[i]), run_esc50.ENROLL_SECONDS) for i in enroll_idx]
    # evaluate_wake draws the enrollment padding from this generator first, as here.
    enrollments, _ = enroll_clips(enroll, np.random.default_rng(seed * 1000 + 5))
    room = np.load(run_mswc.stream_dir(1) / "room_feat.npy")
    streams = {}
    for half in ("H1", "H2"):
        for name in ("eval", "bank"):
            index = common.load_json(folder / f"{half}_{name}_index.json")
            mask = [tuple(s) for s, c in zip(index["spans"], index["category"]) if c == category]
            streams[f"{half}_{name}"] = NegStream(np.load(folder / f"{half}_{name}_feat.npy", mmap_mode="r"),
                                                  np.load(folder / f"{half}_{name}_ends.npy"), mask=mask)
    out = {"category": category, "seed": seed, "shots": shots, "own_half": own_half,
           "negative_hours": {n: s.hours() for n, s in streams.items()}, "methods": {}}
    out["negative_hours"].update({k: main["negative_hours"][k] for k in ("speech_eval", "speech_bank")})
    for method in METHODS:
        if method not in main["methods"]:
            continue
        built = build_wake(enrollments, [room], augment=(method == "nexa"))
        ref = main["methods"][method]
        assert built is not None and abs(built.spec.threshold - ref["threshold"]) < 1e-9, method
        entry = {"pos": ref["pos"]["clean"], "neg": {k: ref["neg"][k] for k in ("speech_eval", "speech_bank")}}
        for name, neg in streams.items():
            scores = fast.wake_score(fast.template_scores(neg.emb, built.spec.templates, built.normalizer))
            scores = np.asarray(scores, dtype=np.float64)
            if neg.mask:
                scores = np.where(mask_frames(neg.mask, neg.ends), scores, np.inf)
            entry["neg"][name] = common.events(scores).tolist()
        out["methods"][method] = entry
    out["seconds"] = round(time.time() - started, 1)
    return out


# ------------------------------------------------------------------ analysis
def first_crossing(rate: np.ndarray, target: float) -> int:
    over = np.nonzero(rate > target + 1e-12)[0]
    return len(rate) - 1 if over.size == 0 else max(0, int(over[0]) - 1)


def analyse() -> None:
    G = common.GRID
    rows = []
    for path in sorted((OUT / "tasks").glob("*.json")):
        r = json.loads(path.read_text())
        h = r["negative_hours"]
        own, other = r["own_half"], ("H2" if r["own_half"] == "H1" else "H1")
        # (bank half, eval half, condition)
        pairs = [(other, other, "seen"), (own, own, "seen"), (own, other, "unseen"), (other, own, "unseen")]
        for method, e in r["methods"].items():
            pos = np.asarray(e["pos"])
            neg = {k: np.asarray(v, dtype=float) for k, v in e["neg"].items()}
            for target in TARGETS_FAH:
                for bank_half, eval_half, cond in pairs:
                    bank = (neg["speech_bank"] + neg[f"{bank_half}_bank"]) / (h["speech_bank"] + h[f"{bank_half}_bank"])
                    idx = first_crossing(bank, target)
                    sound = neg[f"{eval_half}_eval"][idx] / h[f"{eval_half}_eval"]
                    both = (neg["speech_eval"][idx] + neg[f"{eval_half}_eval"][idx]) / (h["speech_eval"]
                                                                                       + h[f"{eval_half}_eval"])
                    rows.append({"category": r["category"], "seed": r["seed"], "shots": r["shots"], "method": method,
                                 "target_fah": target, "condition": cond, "bank_half": bank_half,
                                 "eval_half": eval_half, "eval_is_own_half": eval_half == own,
                                 "tau": float(G[idx]), "frr": float(np.mean(pos > G[idx])), "fah": float(both),
                                 "fah_sound": float(sound),
                                 "fah_speech": float(neg["speech_eval"][idx] / h["speech_eval"]),
                                 "bank_hours": h["speech_bank"] + h[f"{bank_half}_bank"],
                                 "eval_hours": h["speech_eval"] + h[f"{eval_half}_eval"]})
    per = pd.DataFrame(rows)
    if per.empty:
        print("no tasks")
        return
    per.to_csv(OUT / "per_task.csv", index=False)
    rng = np.random.default_rng(0)

    def boot(values: pd.Series, clusters: pd.Series) -> tuple[float, float]:
        g = pd.DataFrame({"v": values.to_numpy(), "c": clusters.to_numpy()}).groupby("c").v.agg(["sum", "count"])
        s, n = g["sum"].to_numpy(), g["count"].to_numpy()
        picks = rng.integers(0, len(s), size=(BOOT, len(s)))
        means = s[picks].sum(1) / n[picks].sum(1)
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    # Average the two (bank, eval) pairs of each condition, then compare conditions on the same wakes.
    task = per.groupby(["category", "seed", "shots", "method", "target_fah", "condition"]).agg(
        frr=("frr", "mean"), fah=("fah", "mean"), fah_sound=("fah_sound", "mean"),
        fah_speech=("fah_speech", "mean")).reset_index()
    out = []
    for (shots, method, target, cond), f in task.groupby(["shots", "method", "target_fah", "condition"]):
        row = {"shots": shots, "method": method, "target_fah": target, "condition": cond, "wakes": len(f),
               "classes": f.category.nunique(), "frr": f.frr.mean()}
        row["frr_lo"], row["frr_hi"] = boot(f.frr, f.category)
        for col in ("fah", "fah_sound", "fah_speech"):
            row[f"{col}_mean"] = f[col].mean()
            row[f"{col}_median"] = float(f[col].median())
        row["fah_lo"], row["fah_hi"] = boot(f.fah, f.category)
        row["share_within_2x"] = float(np.mean(f.fah <= 2 * target + 1e-9))
        out.append(row)
    table = pd.DataFrame(out)
    table.to_csv(OUT / "calibration.csv", index=False)
    paired = []
    for (shots, method, target), f in task.groupby(["shots", "method", "target_fah"]):
        a = f[f.condition == "unseen"].set_index(["category", "seed"])
        b = f[f.condition == "seen"].set_index(["category", "seed"])
        idx = a.index.intersection(b.index)
        if len(idx) < 3:
            continue
        a, b = a.loc[idx], b.loc[idx]
        row = {"shots": shots, "method": method, "target_fah": target, "wakes": len(idx)}
        cats = pd.Series([i[0] for i in idx])
        for col in ("frr", "fah", "fah_sound"):
            d = (a[col] - b[col]).reset_index(drop=True)
            row[f"d_{col}"] = float(d.mean())
            row[f"d_{col}_lo"], row[f"d_{col}_hi"] = boot(d, cats)
        paired.append(row)
    pd.DataFrame(paired).to_csv(OUT / "paired.csv", index=False)
    print(table[table.target_fah == 1.0].to_string(index=False))
    print(pd.DataFrame(paired).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--shots", default="6,3")
    parser.add_argument("--analyse", action="store_true")
    args = parser.parse_args()
    if args.analyse:
        analyse()
        return
    prepare(args.jobs)
    meta, _ = run_esc50.load_clips()
    tasks_dir = OUT / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    work = []
    for shots in [int(s) for s in args.shots.split(",")]:
        for seed in run_esc50.SEEDS:
            for category in sorted(meta.category.unique()):
                path = tasks_dir / f"{category}_seed{seed}_k{shots}.json"
                if not path.is_file() and (MAIN / path.name).is_file():
                    work.append((path, (category, seed, shots)))
    print(f"{len(work)} tasks to run (main-run task files present)", flush=True)
    started = time.time()
    with ProcessPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(run_task, payload): path for path, payload in work}
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            futures[future].write_text(json.dumps(result, ensure_ascii=False))
            print(f"[{done}/{len(work)}] {result['category']} seed {result['seed']} k {result['shots']}: "
                  f"{result['seconds']}s (elapsed {time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
