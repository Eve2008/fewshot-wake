"""Single-thread compute cost on this x86 machine (Raspberry Pi timings are NOT measured).

1. Feature extraction per 80 ms frame (deployed FeatureStream: log-mel ONNX + embedding ONNX).
2. Deployed matcher per frame vs number of templates: TemplateMatcher.step + wake_scores (NumPy) and the full
   WakeDetector.process (the same plus the peak-picking decision layer).
3. The same recurrence compiled with Numba (nexawake/fast.py), per frame.
4. MFCC + DTW baseline: MFCC extraction (librosa, amortised over 60 s blocks) and the compiled DTW at the 10 ms
   MFCC hop, expressed per 80 ms of audio.
5. Bank calibration: scoring the 60-min bank (45,002 frames) for one wake, and counting events on the grid.

Every timing is repeated REPEATS times (interleaved by repeat) so that the spread caused by other jobs on the
shared virtual machine is visible. Per-frame times are saved raw in results/cost/raw_times.npz.

Usage: taskset -c <cpu> python experiments/cost.py
"""

from __future__ import annotations

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[var] = "1"

import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402

OUT = common.RESULTS / "cost"
TEMPLATE_COUNTS = (3, 6, 10, 18, 30, 60, 120, 240)
MFCC_COUNTS = (3, 6, 10)
BANK_COUNTS = (3, 6, 10, 18, 30)
BANK_DEPLOYED = (6, 18)
REPEATS = 3
FEATURE_FRAMES = 2000
MATCH_FRAMES = 1000
HOPS_PER_FRAME = common.CHUNK // common.MFCC_HOP     # 8 MFCC frames per 80 ms
RAW: dict[str, np.ndarray] = {}


def cpu_info() -> dict:
    out = {"cpu": platform.processor(), "hypervisor": None}
    try:
        for line in subprocess.run(["lscpu"], capture_output=True, text=True).stdout.splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "Model name":
                out["cpu"] = value.strip()
            elif key.strip() == "Hypervisor vendor":
                out["hypervisor"] = value.strip()
            elif key.strip() == "CPU(s)":
                out["logical_cpus"] = int(value.strip())
    except (OSError, ValueError):
        pass
    return out


def ms(values) -> float:
    return 1000.0 * float(np.median(values))


def p95(values) -> float:
    return 1000.0 * float(np.percentile(values, 95))


def feature_cost(repeat: int, rng) -> dict:
    m = common.models()
    pcm = common.to_pcm16(rng.normal(0, 2000, (FEATURE_FRAMES + 50) * common.CHUNK)).tobytes()
    stream = m.stream()
    stream.push(pcm[: 50 * common.CHUNK * 2])     # warm-up, fills the 76-frame mel window
    times, walls, mel_t, emb_t = [], [], [], []
    for i in range(50, 50 + FEATURE_FRAMES):
        piece = pcm[i * common.CHUNK * 2: (i + 1) * common.CHUNK * 2]
        w0, t0 = time.perf_counter(), time.thread_time()
        stream.push(piece)
        times.append(time.thread_time() - t0)
        walls.append(time.perf_counter() - w0)
    # Components, measured separately on the same input shapes.
    mel_in = rng.normal(0, 2000, 1760).astype(np.float32)
    win = np.ones((1, 76, 32), dtype=np.float32)
    for _ in range(FEATURE_FRAMES):
        t0 = time.thread_time()
        m.melspectrogram(mel_in)
        mel_t.append(time.thread_time() - t0)
        t0 = time.thread_time()
        m.embed(win)
        emb_t.append(time.thread_time() - t0)
    RAW[f"feature_r{repeat}"] = np.asarray(times)
    RAW[f"feature_wall_r{repeat}"] = np.asarray(walls)
    return {"repeat": repeat, "frames": FEATURE_FRAMES, "feature_ms_median": ms(times), "feature_ms_p95": p95(times),
            "feature_ms_median_wall": ms(walls), "mel_ms_median": ms(mel_t), "embedding_ms_median": ms(emb_t)}


def template_lengths() -> dict:
    """Template lengths of real enrollments (seed 1, k = 10, all 16 target words): the deployed embedding
    templates (80 ms frames) and the MFCC + DTW baseline templates (10 ms frames, as in methods.py)."""

    from methods import enroll_clips
    from nexawake.enroll import build_wake

    plan = common.load_json(common.RESULTS / "mswc" / "plan_seed1.json")
    room = np.load(common.RESULTS / "mswc" / "streams" / "seed1" / "room_feat.npy")
    lengths, mfcc_lengths = [], []
    for task in plan:
        clips = [common.read(link) for link in task["enroll"]]
        enrollments, audio = enroll_clips(clips, np.random.default_rng(1))
        built = build_wake(enrollments, [room], augment=False)
        if built is not None:
            lengths += [len(t) for t in built.spec.templates]
        for padded, onset, offset in audio:
            frames = common.mfcc(padded).shape[0]
            first = max(0, onset // common.MFCC_HOP)
            last = min(frames, max(onset // common.MFCC_HOP + 3, offset // common.MFCC_HOP))
            mfcc_lengths.append(last - first)
    return {"embedding": lengths, "mfcc": mfcc_lengths}


def matcher_cost(count: int, length: int, repeat: int, rng) -> dict:
    from nexawake import fast
    from nexawake.wake import Normalizer, TemplateMatcher, WakeDetector, WakeSpec

    dim = 128
    templates = [rng.normal(size=(length, dim)).astype(np.float32) for _ in range(count)]
    queries = rng.normal(size=(MATCH_FRAMES, dim)).astype(np.float32)
    normalizer = Normalizer.fit([queries])
    spec = WakeSpec("w", "w", templates, 0.3)
    matcher = TemplateMatcher([spec], normalizer)
    detector = WakeDetector([spec], normalizer)
    for row in queries[:20]:
        matcher.wake_scores(matcher.step(row))
        detector.process(row)
    matcher.reset()
    detector.reset()
    times, walls, det = [], [], []
    for row in queries:
        w0, t0 = time.perf_counter(), time.thread_time()
        matcher.wake_scores(matcher.step(row))
        times.append(time.thread_time() - t0)
        walls.append(time.perf_counter() - w0)
    for row in queries:
        t0 = time.thread_time()
        detector.process(row)
        det.append(time.thread_time() - t0)
    fast.template_scores(queries[:50], templates, normalizer)   # compile (cached after the first call)
    t0 = time.thread_time()
    fast.wake_score(fast.template_scores(queries, templates, normalizer))
    compiled = (time.thread_time() - t0) / MATCH_FRAMES
    RAW[f"matcher_n{count}_l{length}_r{repeat}"] = np.asarray(times)
    RAW[f"detector_n{count}_l{length}_r{repeat}"] = np.asarray(det)
    return {"templates": count, "length": length, "repeat": repeat, "frames": MATCH_FRAMES,
            "deployed_ms_median": ms(times), "deployed_ms_p95": p95(times), "deployed_ms_median_wall": ms(walls),
            "detector_ms_median": ms(det), "detector_ms_p95": p95(det), "compiled_ms": 1000 * compiled}


def mfcc_feature_cost(repeat: int, rng) -> dict:
    seconds = 60.0
    values = rng.normal(0, 2000, int(seconds * common.SR)).astype(np.float32)
    common.mfcc(values[: common.SR])         # warm-up
    t0 = time.thread_time()
    common.mfcc(values)
    elapsed = time.thread_time() - t0
    frames = seconds / common.FRAME_SECONDS
    return {"repeat": repeat, "seconds": seconds, "mfcc_ms_per_80ms": 1000 * elapsed / frames}


def mfcc_dtw_cost(count: int, length: int, repeat: int, rng) -> dict:
    from nexawake import fast

    hops = MATCH_FRAMES * HOPS_PER_FRAME
    templates = [rng.normal(size=(length, 39)).astype(np.float32) for _ in range(count)]
    queries = rng.normal(size=(hops, 39)).astype(np.float32)
    normalizer = common.fit_normalizer([queries])
    fast.template_scores(queries[:50], templates, normalizer)
    t0 = time.thread_time()
    fast.wake_score(fast.template_scores(queries, templates, normalizer))
    elapsed = time.thread_time() - t0
    return {"templates": count, "length": length, "repeat": repeat,
            "compiled_ms_per_80ms": 1000 * elapsed / MATCH_FRAMES}


def bank_cost(count: int, length: int, repeat: int, rng, deployed: bool) -> dict:
    from nexawake import fast
    from nexawake.wake import Normalizer, TemplateMatcher, WakeSpec

    bank = np.load(common.RESULTS / "bank" / "clean_feat.npy").astype(np.float32)
    templates = [bank[i * 50: i * 50 + length] + 0.05 * rng.normal(size=(length, 128)).astype(np.float32)
                 for i in range(count)]
    normalizer = Normalizer.fit([bank])
    t0 = time.thread_time()
    scores = fast.wake_score(fast.template_scores(bank, templates, normalizer))
    compiled = time.thread_time() - t0
    t0 = time.thread_time()
    events = common.events(scores)
    counting = time.thread_time() - t0
    out = {"templates": count, "length": length, "repeat": repeat, "bank_frames": int(bank.shape[0]),
           "compiled_seconds": round(compiled, 3), "event_count_seconds": round(counting, 3),
           "grid_thresholds": int(common.GRID.size)}
    if deployed:
        matcher = TemplateMatcher([WakeSpec("w", "w", templates, 0.3)], normalizer)
        t0 = time.thread_time()
        for row in bank:
            matcher.wake_scores(matcher.step(row))
        out["deployed_seconds"] = round(time.thread_time() - t0, 2)
    out["events_check"] = int(events[-1])
    return out


def summarise(frame: pd.DataFrame, keys: list[str], cols: list[str]) -> pd.DataFrame:
    """Median over repeats, plus the smallest and largest repeat (spread caused by the shared machine)."""

    agg = {}
    for col in cols:
        agg[col] = (col, "median")
        agg[f"{col}_min"] = (col, "min")
        agg[f"{col}_max"] = (col, "max")
    return frame.groupby(keys).agg(**agg).reset_index()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    import numba
    import onnxruntime

    info = {**cpu_info(), "python": platform.python_version(), "numpy": np.__version__,
            "onnxruntime": onnxruntime.__version__, "numba": numba.__version__,
            "load_average_before": [round(v, 2) for v in os.getloadavg()],
            "affinity": sorted(os.sched_getaffinity(0)), "repeats": REPEATS,
            "clock": "thread CPU time (time.thread_time); wall time in *_wall columns",
            "note": ("Single thread pinned to one vCPU of a VMware virtual machine on a shared desktop CPU with "
                     "other jobs running; Raspberry Pi 4 timings not measured.")}
    print(info, flush=True)
    t_start = time.time()
    lengths = template_lengths()
    typical = int(np.median(lengths["embedding"]))
    typical_mfcc = int(np.median(lengths["mfcc"]))
    info["template_length_median"] = typical
    info["template_length_p90"] = int(np.percentile(lengths["embedding"], 90))
    info["template_length_max"] = int(np.max(lengths["embedding"]))
    info["template_lengths_n"] = len(lengths["embedding"])
    info["mfcc_template_length_median"] = typical_mfcc
    info["mfcc_template_length_p90"] = int(np.percentile(lengths["mfcc"], 90))
    info["mfcc_template_lengths_n"] = len(lengths["mfcc"])
    pd.DataFrame({"kind": ["embedding"] * len(lengths["embedding"]) + ["mfcc"] * len(lengths["mfcc"]),
                  "frames": lengths["embedding"] + lengths["mfcc"]}).to_csv(OUT / "template_lengths.csv", index=False)
    print("template lengths", typical, typical_mfcc, flush=True)

    feats, match, mf_feat, mf_dtw, banks = [], [], [], [], []
    for repeat in range(REPEATS):
        rng = np.random.default_rng(100 + repeat)
        feats.append(feature_cost(repeat, rng))
        print(feats[-1], flush=True)
        for length in (typical, 30):
            for count in TEMPLATE_COUNTS:
                match.append(matcher_cost(count, length, repeat, rng))
                print(match[-1], flush=True)
        mf_feat.append(mfcc_feature_cost(repeat, rng))
        for count in MFCC_COUNTS:
            mf_dtw.append(mfcc_dtw_cost(count, typical_mfcc, repeat, rng))
        print(mf_feat[-1], mf_dtw[-len(MFCC_COUNTS):], flush=True)
        for count in BANK_COUNTS:
            banks.append(bank_cost(count, typical, repeat, rng, deployed=count in BANK_DEPLOYED and repeat == 0))
            print(banks[-1], flush=True)
        # Raw files are rewritten after every repeat, so an interruption loses at most one repeat.
        pd.DataFrame(feats).to_csv(OUT / "raw_features.csv", index=False)
        pd.DataFrame(match).to_csv(OUT / "raw_matcher.csv", index=False)
        pd.DataFrame(mf_feat).to_csv(OUT / "raw_mfcc_features.csv", index=False)
        pd.DataFrame(mf_dtw).to_csv(OUT / "raw_mfcc_dtw.csv", index=False)
        pd.DataFrame(banks).to_csv(OUT / "raw_bank_calibration.csv", index=False)
        np.savez_compressed(OUT / "raw_times.npz", **RAW)

    f = pd.DataFrame(feats)
    feature_ms = float(f.feature_ms_median.median())
    info["features"] = {
        "frames_per_repeat": FEATURE_FRAMES,
        "feature_ms_median": feature_ms,
        "feature_ms_median_min": float(f.feature_ms_median.min()),
        "feature_ms_median_max": float(f.feature_ms_median.max()),
        "feature_ms_p95": float(f.feature_ms_p95.median()),
        "feature_ms_median_wall": float(f.feature_ms_median_wall.median()),
        "mel_ms_median": float(f.mel_ms_median.median()),
        "embedding_ms_median": float(f.embedding_ms_median.median()),
    }
    table = summarise(pd.DataFrame(match), ["templates", "length"],
                      ["deployed_ms_median", "deployed_ms_p95", "deployed_ms_median_wall", "detector_ms_median",
                       "detector_ms_p95", "compiled_ms"])
    table["total_ms_with_features"] = table.detector_ms_median + feature_ms
    table["real_time_factor"] = table.total_ms_with_features / (common.FRAME_SECONDS * 1000)
    table["total_ms_compiled"] = table.compiled_ms + feature_ms
    table.to_csv(OUT / "matcher_cost.csv", index=False)

    mf = pd.DataFrame(mf_feat)
    mfcc_ms = float(mf.mfcc_ms_per_80ms.median())
    info["mfcc_features"] = {"mfcc_ms_per_80ms": mfcc_ms, "mfcc_ms_per_80ms_min": float(mf.mfcc_ms_per_80ms.min()),
                             "mfcc_ms_per_80ms_max": float(mf.mfcc_ms_per_80ms.max()),
                             "block_seconds": float(mf.seconds.iloc[0])}
    md = summarise(pd.DataFrame(mf_dtw), ["templates", "length"], ["compiled_ms_per_80ms"])
    md["total_ms_per_80ms"] = md.compiled_ms_per_80ms + mfcc_ms
    md.to_csv(OUT / "mfcc_dtw_cost.csv", index=False)

    b = pd.DataFrame(banks)
    bt = summarise(b, ["templates", "length", "bank_frames", "grid_thresholds"],
                   ["compiled_seconds", "event_count_seconds"])
    dep = b.dropna(subset=["deployed_seconds"])[["templates", "deployed_seconds"]] if "deployed_seconds" in b else None
    if dep is not None and not dep.empty:
        bt = bt.merge(dep, on="templates", how="left")
    bt.to_csv(OUT / "bank_calibration_cost.csv", index=False)

    info["load_average_after"] = [round(v, 2) for v in os.getloadavg()]
    info["run_seconds"] = round(time.time() - t_start, 1)
    common.dump_json(OUT / "cost_info.json", info)
    print(info, flush=True)
    print(table.to_string(), flush=True)
    print(md.to_string(), flush=True)
    print(bt.to_string(), flush=True)


if __name__ == "__main__":
    main()
