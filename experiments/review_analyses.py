"""Analyses added after review. Reads per_task.csv files, raw task files and the new result folders; does
not rescore anything. Every number of the revised manuscript that is not in summary.md comes from here.

Writes results/review/:
  ablation_mswc.csv         paired method differences, cross-speaker (2 x 2 design of templates x features and
                            alignment, few-shot heads, SSL + DTW), per k x condition x rule x target
  ablation_samespk.csv      the same for the same-speaker sets (samespk_v2: own clips masked from the stream)
  shots_new.csv             k = 10 vs 3 and 10 vs 6 for the new variants (cross-speaker)
  rules_paired.csv          tail fit vs order statistic / empirical / oracle, paired, cross-speaker
  dev_split.csv             rule and fit range chosen on seed 1 only, then reported on seeds 2-3
  seed_spread.csv           per-seed means and a two-way (word x seed) bootstrap for the main cells
  tail_models.csv           exponential and generalised Pareto maximum-likelihood tail fits vs the OLS fit
  tail_shape.csv            per-wake exponential slope on the bank vs on the evaluation negatives
  enrollment.csv            recordings rejected by the energy rule per wake
  enroll_speaker_overlap.csv clips of a wake's enrollment speakers inside its 4-h negative stream
  esc50_tail.csv            sound wakes with the tail fit on the speech + sound bank, FA/h split by stream
  samespk_own.csv           own-voice false alarms of the same-speaker wakes: alarm counts, hours, pooled FA/h
                            with cluster-bootstrap intervals over speakers and over words, per-wake resolution
  samespk_words.csv         the same-speaker results split by repeated word (main set)
  samespk_own_curve.csv     same speaker, one threshold per set and k chosen on the pooled own-voice stream:
                            FRR at matched own-voice FA/h (an oracle reference)
  poisson_reference.csv     worst wake and share of wakes within 2 x target expected from Poisson sampling
                            alone, against the observed values (cross-speaker, 4-h streams)
  summary_review.md         compact text summary of all of the above
Usage: python experiments/review_analyses.py
"""

from __future__ import annotations

import collections
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from summarize import first_crossing, order_statistic_index, tail_index  # noqa: E402

R = common.RESULTS
OUT = R / "review"
G = common.GRID
BOOT = 2000
RULES = ["deployed", "global-bank", "bank60", "bank60-os", "bank60-tail", "oracle-wake", "oracle-global"]
PAIRS = [
    ("nexa-noaug", "nexa", "noisy copies removed (full frame)"),
    ("emb-only", "nexa", "log-mel removed (noisy templates)"),
    ("emb-only-noaug", "nexa-noaug", "log-mel removed (clean templates)"),
    ("emb-only-noaug", "emb-only", "noisy copies removed (embedding only)"),
    ("mel-only", "nexa", "embedding removed (noisy templates)"),
    ("mel-only-noaug", "nexa-noaug", "embedding removed (clean templates)"),
    ("prototype", "nexa", "DTW removed (noisy templates)"),
    ("prototype-noaug", "nexa-noaug", "DTW removed (clean templates)"),
    ("prototype-noaug", "prototype", "noisy copies removed (prototype)"),
    ("nexa-noaug-bnorm", "nexa-noaug", "bank normaliser"),
    ("nexa-nonorm", "nexa", "normaliser removed"),
    ("nexa-min", "nexa", "best single template"),
    ("emb-head", "nexa-noaug", "logistic head vs clean templates"),
    ("emb-head", "emb-only-noaug", "logistic head vs embedding-only DTW (clean)"),
    ("emb-head-aug", "emb-head", "noisy copies as head positives"),
    ("ssl-dtw", "nexa-noaug", "mHuBERT-147 + DTW vs clean templates"),
    ("ssl-dtw", "mfcc-dtw", "mHuBERT-147 + DTW vs MFCC + DTW"),
    ("mfcc-dtw", "nexa-noaug", "MFCC + DTW vs clean templates"),
    ("ssl-dtw-lc", "ssl-dtw", "mHuBERT-147: limited vs offline context"),
    ("ssl-dtw-lc", "nexa-noaug", "mHuBERT-147 (limited context) vs clean templates"),
]


def stable_rng(*parts) -> np.random.Generator:
    """A generator seeded from the comparison itself, so that every interval is the same whatever else the
    script computes before it (adding a contrast does not move the other intervals)."""

    return np.random.default_rng(zlib.crc32(repr(parts).encode("utf-8")))


def boot(diff: np.ndarray, clusters: np.ndarray, rng) -> tuple[float, float]:
    frame = pd.DataFrame({"d": diff, "c": clusters})
    g = frame.groupby("c").d.agg(["sum", "count"])
    s, n = g["sum"].to_numpy(), g["count"].to_numpy()
    picks = rng.integers(0, len(s), size=(BOOT, len(s)))
    means = s[picks].sum(1) / n[picks].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(frame: pd.DataFrame, col: str, a, b, keys: list[str], rng, extra_cols=()) -> dict | None:
    x = frame[frame[col] == a].set_index(keys)
    y = frame[frame[col] == b].set_index(keys)
    idx = x.index.intersection(y.index)
    if len(idx) < 3:
        return None
    x, y = x.loc[idx], y.loc[idx]
    rng = stable_rng(col, a, b, tuple(map(str, idx)), round(float(x.frr.sum()), 6), round(float(y.frr.sum()), 6))
    row = {"a": a, "b": b, "wakes": len(idx), "clusters": int(x.cluster.nunique()),
           "frr_a": float(x.frr.mean()), "frr_b": float(y.frr.mean()),
           "fah_a": float(x.fah.mean()), "fah_b": float(y.fah.mean())}
    for metric in ("frr", "fah") + tuple(extra_cols):
        d = (x[metric] - y[metric]).to_numpy()
        lo, hi = boot(d, x.cluster.to_numpy(), rng)
        row[f"d_{metric}"], row[f"d_{metric}_lo"], row[f"d_{metric}_hi"] = float(d.mean()), lo, hi
        row[f"{metric}_a_better"] = int(np.sum(d < 0))
        row[f"{metric}_b_better"] = int(np.sum(d > 0))
    for c in extra_cols:
        row[f"{c}_a"], row[f"{c}_b"] = float(x[c].mean()), float(y[c].mean())
    return row


def read(name: str) -> pd.DataFrame:
    path = R / name / "per_task.csv"
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def mswc_all() -> pd.DataFrame:
    parts = [read("mswc"), read("mswc_x"), read("ssl"), read("ssl_lc")]
    per = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    return per[per.rule.isin(RULES)]


def ablation_mswc(per, rng) -> pd.DataFrame:
    rows = []
    for (k, cond, rule, target), frame in per.groupby(["shots", "condition", "rule", "target_fah"]):
        for a, b, label in PAIRS:
            row = compare(frame, "method", a, b, ["seed", "word"], rng)
            if row:
                rows.append({"shots": k, "condition": cond, "rule": rule, "target_fah": target, "contrast": label, **row})
    return pd.DataFrame(rows)


def ablation_samespk(rng) -> pd.DataFrame:
    same = read("samespk_v2")
    if same.empty:
        return pd.DataFrame()
    ssl = read("ssl_samespk")
    if not ssl.empty:
        same = pd.concat([same, ssl], ignore_index=True)
    same = same[same.rule.isin(RULES)]
    rows = []
    for (setname, k, cond, rule, target), frame in same.groupby(["set", "shots", "condition", "rule", "target_fah"]):
        for a, b, label in PAIRS:
            row = compare(frame, "method", a, b, ["index"], rng, extra_cols=("fah_own_clean",))
            if row:
                rows.append({"set": setname, "shots": k, "condition": cond, "rule": rule, "target_fah": target,
                             "contrast": label, **row})
    return pd.DataFrame(rows)


def shots_new(per, rng) -> pd.DataFrame:
    rows = []
    for (method, cond, rule, target), frame in per.groupby(["method", "condition", "rule", "target_fah"]):
        for a, b in ((10, 3), (10, 6), (6, 3)):
            row = compare(frame, "shots", a, b, ["seed", "word"], rng)
            if row:
                rows.append({"method": method, "condition": cond, "rule": rule, "target_fah": target, **row})
    return pd.DataFrame(rows)


def rules_paired(rng) -> pd.DataFrame:
    per = read("mswc")
    rows = []
    for (k, method, cond, target), frame in per.groupby(["shots", "method", "condition", "target_fah"]):
        if method not in ("nexa", "nexa-noaug", "mfcc-dtw"):
            continue
        for a, b in (("bank60-tail", "bank60-os"), ("bank60-tail", "bank60"), ("bank60-os", "oracle-wake"),
                     ("bank60-tail", "oracle-wake"), ("bank60-tail", "global-bank")):
            row = compare(frame, "rule", a, b, ["seed", "word"], rng)
            if row:
                rows.append({"shots": k, "method": method, "condition": cond, "target_fah": target, **row})
    return pd.DataFrame(rows)


def dev_split() -> pd.DataFrame:
    """Choose rule and fit range on seed 1; report on seeds 2-3. Criterion (fixed in advance): the lowest
    mean FRR among the rules whose mean FA/h on the seed-1 negatives is at most the target."""

    per = read("mswc")
    per = per[(per.condition == "clean") & per.method.isin(["nexa-noaug", "nexa"])]
    cand = ["bank60", "bank60-os", "bank60-tail", "bank60-tail[1,20]", "bank60-tail[3,30]", "bank60-tail[2,100]",
            "bank60-tail[5,200]", "bank30-tail", "global-bank"]
    rows = []
    for (method, target), frame in per[per.rule.isin(cand)].groupby(["method", "target_fah"]):
        dev = frame[frame.seed == 1].groupby("rule").agg(frr=("frr", "mean"), fah=("fah", "mean"))
        ok = dev[dev.fah <= target + 1e-9]
        chosen = ok.frr.idxmin() if len(ok) else dev.fah.idxmin()
        test = frame[frame.seed.isin([2, 3])].groupby("rule").agg(
            frr=("frr", "mean"), fah=("fah", "mean"), fah_max=("fah", "max"),
            within2=("fah", lambda x: float(np.mean(x <= 2 * target + 1e-9))))
        for rule in cand:
            if rule not in dev.index:
                continue
            rows.append({"method": method, "target_fah": target, "rule": rule, "chosen_on_seed1": rule == chosen,
                         "dev_frr": dev.loc[rule, "frr"], "dev_fah": dev.loc[rule, "fah"],
                         "test_frr": test.loc[rule, "frr"], "test_fah": test.loc[rule, "fah"],
                         "test_fah_max": test.loc[rule, "fah_max"], "test_within_2x": test.loc[rule, "within2"]})
    return pd.DataFrame(rows)


def seed_spread(per) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    sel = per[(per.condition == "clean") & (per.target_fah == 1.0) & per.rule.isin(["bank60-tail", "oracle-wake", "deployed"])]
    for (method, k, rule), frame in sel.groupby(["method", "shots", "rule"]):
        rng = stable_rng("seed_spread", method, int(k), rule)
        by_seed = frame.groupby("seed").agg(frr=("frr", "mean"), fah=("fah", "mean"))
        words = sorted(frame.word.unique())
        seeds = sorted(frame.seed.unique())
        cell = frame.set_index(["word", "seed"])
        frr = cell.frr.unstack().reindex(index=words, columns=seeds).to_numpy()
        fah = cell.fah.unstack().reindex(index=words, columns=seeds).to_numpy()
        bw = rng.integers(0, len(words), size=(BOOT, len(words)))
        bs = rng.integers(0, len(seeds), size=(BOOT, len(seeds)))
        def two_way(m):
            vals = np.array([np.nanmean(m[np.ix_(bw[i], bs[i])]) for i in range(BOOT)])
            return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
        f_lo, f_hi = two_way(frr)
        a_lo, a_hi = two_way(fah)
        row = {"method": method, "shots": k, "rule": rule, "frr": float(np.nanmean(frr)), "fah": float(np.nanmean(fah)),
               "frr_2way_lo": f_lo, "frr_2way_hi": f_hi, "fah_2way_lo": a_lo, "fah_2way_hi": a_hi}
        for s in seeds:
            row[f"frr_seed{s}"], row[f"fah_seed{s}"] = by_seed.loc[s, "frr"], by_seed.loc[s, "fah"]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- tail models
def pseudo_scores(counts: np.ndarray, upto: int) -> np.ndarray:
    """Event 'scores' from a cumulative count curve: each increment at grid index g adds events at G[g]."""

    inc = np.diff(np.concatenate([[0], counts[: upto + 1]]))
    inc = np.maximum(inc, 0)
    return np.repeat(G[: upto + 1], inc.astype(int))


def tail_threshold(counts: np.ndarray, hours: float, target: float, model: str, hi: int = 50) -> int:
    over = np.nonzero(counts > hi)[0]
    stop = int(over[0]) - 1 if over.size else len(counts) - 1
    n = int(counts[stop])
    if n < 3 or np.unique(counts[: stop + 1][counts[: stop + 1] >= 2]).size < 3:
        return first_crossing(counts / hours, target)
    u = float(G[stop])
    y = u - pseudo_scores(counts, stop)
    y = y[y >= 0]
    need = target * hours
    if n <= need:
        return stop
    if model == "exp-mle":
        lam = len(y) / max(float(y.sum()), 1e-9)
        ystar = np.log(n / need) / lam
    else:
        from scipy.stats import genpareto

        jitter = y + 1e-4          # grid ties at 0 make the likelihood degenerate
        xi, _loc, sigma = genpareto.fit(jitter, floc=0.0)
        if abs(xi) < 1e-6:
            ystar = sigma * np.log(n / need)
        else:
            ystar = sigma / xi * ((n / need) ** xi - 1.0)
    tau = u - ystar
    return int(np.clip(np.searchsorted(G, tau, side="right") - 1, 0, len(G) - 1))


def tail_models() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, shape = [], []
    for path in sorted((R / "mswc" / "tasks").glob("*.json")):
        r = json.loads(path.read_text())
        for method in ("nexa-noaug", "nexa"):
            e = r["methods"].get(method)
            if e is None:
                continue
            bank = np.asarray(e["neg"]["bank_clean"], dtype=float)
            hb = r["negative_hours"]["bank_clean"]
            ev = np.asarray(e["neg"]["eval_clean"], dtype=float)
            he = r["negative_hours"]["eval_clean"]
            pos = np.asarray(e["pos"]["clean"])
            for target in (0.5, 1.0, 2.0):
                idx = {"ols": tail_index(bank, hb, target)[0],
                       "exp-mle": tail_threshold(bank, hb, target, "exp-mle"),
                       "gpd-mle": tail_threshold(bank, hb, target, "gpd-mle")}
                for model, i in idx.items():
                    rows.append({"seed": r["seed"], "shots": r["shots"], "word": r["word"], "method": method,
                                 "target_fah": target, "model": model, "tau": float(G[i]),
                                 "frr": float(np.mean(pos > G[i])), "fah": float(ev[i] / he)})
            # Exponential slope of ln N(tau) on the bank (2..50 events) and on the evaluation stream over the
            # same range of rates (8..200 events in 4 h, i.e. 2..50 per hour).
            def slope(counts, lo, hi):
                over = np.nonzero(counts > hi)[0]
                stop = int(over[0]) if over.size else len(counts)
                sel = np.nonzero(counts[:stop] >= lo)[0]
                if np.unique(counts[sel]).size < 3:
                    return np.nan
                return float(np.polyfit(G[sel], np.log(counts[sel]), 1)[0])
            shape.append({"seed": r["seed"], "shots": r["shots"], "word": r["word"], "method": method,
                          "slope_bank": slope(bank, 2, 50), "slope_eval": slope(ev, 8, 200),
                          "slope_eval_low": slope(ev, 2, 50)})
    return pd.DataFrame(rows), pd.DataFrame(shape)


# ---------------------------------------------------------------- misc counts
def enrollment_counts() -> pd.DataFrame:
    rows = []
    for path in sorted((R / "mswc" / "tasks").glob("*.json")):
        r = json.loads(path.read_text())
        rows.append({"seed": r["seed"], "shots": r["shots"], "word": r["word"], "enrolled": r["enrolled"],
                     "rejected": r["shots"] - r["enrolled"]})
    return pd.DataFrame(rows)


def enroll_overlap() -> pd.DataFrame:
    split = common.load_json(R / "splits.json")
    rows = []
    for seed in ("1", "2", "3"):
        neg = collections.Counter(split["seeds"][seed]["negative_speakers"])
        for t in common.load_json(R / "mswc" / f"plan_seed{seed}.json"):
            for k in (3, 6, 10):
                rows.append({"seed": int(seed), "shots": k, "word": t["word"],
                             "enroll_speaker_clips": int(sum(neg[s] for s in t["enroll_speakers"][:k])),
                             "positive_speaker_clips": int(sum(neg[s] for s in t["positive_speakers"])),
                             "clips": len(split["seeds"][seed]["negative_speakers"])})
    return pd.DataFrame(rows)


def esc50_tail() -> pd.DataFrame:
    cal = pd.read_csv(R / "esc50" / "calibration.csv")
    keep = cal[cal.rule.isin(["speech+soundbank", "speech+soundbank-tail", "speechbank-tail", "oracle-wake"])
               & cal.method.isin(["nexa-noaug", "mfcc-dtw", "nexa"]) & (cal.condition == "clean")]
    cols = [c for c in ["shots", "method", "rule", "target_fah", "wakes", "frr", "frr_lo", "frr_hi", "fah_mean",
                        "fah_sound_eval_pooled", "fah_speech_eval_pooled", "fah_sound_eval_mean"] if c in keep.columns]
    return keep[cols].sort_values(["shots", "method", "target_fah", "rule"])


# ---------------------------------------------------------------- same-speaker own-voice alarms
def _ratio_boot(alarms: np.ndarray, hours: np.ndarray, clusters: np.ndarray, rng) -> tuple[float, float]:
    frame = pd.DataFrame({"a": alarms, "h": hours, "c": clusters}).groupby("c")[["a", "h"]].sum()
    a, h = frame.a.to_numpy(), frame.h.to_numpy()
    rng = stable_rng(tuple(map(str, frame.index)), tuple(a.tolist()), round(float(h.sum()), 6))
    picks = rng.integers(0, len(a), size=(BOOT, len(a)))
    ratios = a[picks].sum(1) / h[picks].sum(1)
    return float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def _samespk_frame() -> pd.DataFrame:
    same = read("samespk_v2")
    if same.empty:
        return same
    ssl = read("ssl_samespk")
    if not ssl.empty:
        same = pd.concat([same, ssl], ignore_index=True)
    same = same[(same.condition == "clean") & (same.target_fah == 1.0)
                & same.rule.isin(["deployed", "bank60-tail", "bank60-os", "oracle-wake"])].copy()
    same["own_alarms"] = np.rint(same.fah_own_clean * same.hours_own_clean).astype(int)
    return same


def samespk_own(rng) -> pd.DataFrame:
    """Own-voice alarms with intervals: one alarm on a speaker's ~4 min of own speech adds ~15 FA/h to that
    wake, so the pooled rate rests on few alarms. Bootstrap of the pooled ratio (alarms / hours) over
    speakers (= wakes) and over the repeated words."""

    same = _samespk_frame()
    rows = []
    for (setname, k, method, rule), f in same.groupby(["set", "shots", "method", "rule"]):
        a, h = f.own_alarms.to_numpy(), f.hours_own_clean.to_numpy()
        s_lo, s_hi = _ratio_boot(a, h, f.speaker.to_numpy(), rng)
        w_lo, w_hi = _ratio_boot(a, h, f.word.to_numpy(), rng)
        rows.append({"set": setname, "shots": k, "method": method, "rule": rule, "wakes": len(f),
                     "speakers": int(f.speaker.nunique()), "words": int(f.word.nunique()),
                     "frr": float(f.frr.mean()), "fah_other": float(f.fah.mean()),
                     "own_hours": float(h.sum()), "own_alarms": int(a.sum()),
                     "own_fah_pooled": float(a.sum() / h.sum()), "own_fah_spk_lo": s_lo, "own_fah_spk_hi": s_hi,
                     "own_fah_word_lo": w_lo, "own_fah_word_hi": w_hi,
                     "wakes_with_own_alarm": int(np.sum(a > 0)), "own_alarms_max_wake": int(a.max()),
                     "own_minutes_median": float(np.median(h) * 60),
                     "resolution_fah_median": float(np.median(1.0 / h)),
                     "own_fah_over_other": float(a.sum() / h.sum() / max(f.fah.mean(), 1e-9))})
    return pd.DataFrame(rows)


def samespk_words() -> pd.DataFrame:
    same = _samespk_frame()
    if same.empty:
        return same
    same = same[(same.set == "main") & same.rule.isin(["bank60-tail", "deployed"])]
    top = same[(same.method == "nexa-noaug") & (same.shots == 10) & (same.rule == "bank60-tail")]
    common_words = top.word.value_counts()
    keep = set(common_words[common_words >= 5].index)
    same = same.assign(word_group=np.where(same.word.isin(keep), same.word, "other words"))
    rows = []
    for (k, method, rule, group), f in same.groupby(["shots", "method", "rule", "word_group"]):
        rows.append({"shots": k, "method": method, "rule": rule, "word": group, "wakes": len(f),
                     "frr": float(f.frr.mean()), "fah_other": float(f.fah.mean()),
                     "own_alarms": int(f.own_alarms.sum()), "own_hours": float(f.hours_own_clean.sum()),
                     "own_fah_pooled": float(f.own_alarms.sum() / f.hours_own_clean.sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Poisson reference for the spread over wakes
def poisson_reference(rng) -> pd.DataFrame:
    """If every wake had the same true rate, how large would the worst wake and how small the share within
    2 x target be from the sampling noise of one 4-h stream alone? Compared with the observed values; the
    excess dispersion (variance of counts minus their mean) estimates the spread of the true per-wake rates."""

    from scipy.stats import chi2

    per = read("mswc")
    hours = {}
    for path in sorted((R / "mswc" / "tasks").glob("seed*_k10_w00.json")):
        r = json.loads(path.read_text())
        hours[int(r["seed"])] = float(r["negative_hours"]["eval_clean"])
    per = per[(per.condition == "clean") & (per.target_fah == 1.0) & per.method.isin(["nexa", "nexa-noaug"])
              & per.rule.isin(["bank60-tail", "bank60-os", "global-bank", "oracle-wake"])].copy()
    per["hours"] = per.seed.map(hours)
    per["count"] = np.rint(per.fah * per.hours).astype(int)
    sims = 20000
    rows = []
    for (method, rule, k), f in list(per.groupby(["method", "rule", "shots"])) + \
            [((m, r, "all"), g) for (m, r), g in per.groupby(["method", "rule"])]:
        c, h = f["count"].to_numpy(), f.hours.to_numpy()
        n, mean = len(c), float(c.mean())
        row = {"method": method, "rule": rule, "shots": k, "wakes": n, "hours_per_wake": float(np.median(h)),
               "mean_fah": float((c / h).mean()), "obs_worst_fah": float((c / h).max()),
               "obs_share_within_2": float(np.mean(c / h <= 2.0 + 1e-9)),
               "dispersion_index": float(c.var(ddof=1) / mean) if mean > 0 else np.nan,
               "chi2_p": float(chi2.sf(((c - mean) ** 2).sum() / mean, n - 1)) if mean > 0 else np.nan,
               "rate_cv": float(np.sqrt(max(0.0, c.var(ddof=1) - mean)) / mean) if mean > 0 else np.nan}
        for label, lam in (("at_mean", mean), ("at_target", 1.0 * float(np.median(h)))):
            rng = stable_rng(method, rule, str(k), label)
            draws = rng.poisson(lam, size=(sims, n)) / np.median(h)
            worst = draws.max(axis=1)
            share = (draws <= 2.0 + 1e-9).mean(axis=1)
            row.update({f"poisson_{label}_worst_median": float(np.median(worst)),
                        f"poisson_{label}_worst_p95": float(np.percentile(worst, 95)),
                        f"poisson_{label}_share_within_2": float(share.mean()),
                        f"poisson_{label}_share_within_2_p05": float(np.percentile(share, 5)),
                        f"poisson_{label}_p_worst_ge_obs": float(np.mean(worst >= row["obs_worst_fah"] - 1e-9))})
        rows.append(row)
    return pd.DataFrame(rows)


def samespk_own_curve() -> pd.DataFrame:
    """Trade-off within one speaker: one threshold for all wakes of a set and k, chosen on the pooled own-voice
    stream (an oracle, not deployable), so that variants are compared at the same own-voice FA/h. Reports FRR
    and the mean other-speaker FA/h at the largest threshold whose pooled own-voice rate is at most the target."""

    methods = ("nexa", "nexa-noaug", "emb-only-noaug", "mfcc-dtw", "ssl-dtw")
    curves = {}
    for folder in ("samespk_v2", "ssl_samespk"):
        for path in sorted((R / folder / "tasks").glob("*.json")):
            r = json.loads(path.read_text())
            for method, e in r.get("methods", {}).items():
                if method not in methods:
                    continue
                pos = np.asarray(e["pos"]["clean"])
                c = curves.setdefault((r["set"], r["shots"], method), {"frr": [], "own": [], "own_h": 0.0, "other": []})
                c["frr"].append((pos[None, :] > G[:, None]).mean(axis=1))
                c["own"].append(np.asarray(e["neg"]["own_clean"], dtype=float))
                c["own_h"] += r["negative_hours"]["own_clean"]
                c["other"].append(np.asarray(e["neg"]["eval_clean"], dtype=float) / r["negative_hours"]["eval_clean"])
    rows = []
    for (setname, k, method), c in sorted(curves.items()):
        frr = np.mean(c["frr"], axis=0)
        own = np.sum(c["own"], axis=0) / c["own_h"]
        other = np.mean(c["other"], axis=0)
        for target in (1.0, 2.0, 5.0, 10.0):
            i = first_crossing(own, target)
            rows.append({"set": setname, "shots": k, "method": method, "wakes": len(c["frr"]),
                         "own_target_fah": target, "tau": float(G[i]), "frr": float(frr[i]),
                         "own_fah_pooled": float(own[i]), "other_fah_mean": float(other[i])})
    return pd.DataFrame(rows)


def fmt_pair(row) -> str:
    return (f"{row.d_frr:+.3f} [{row.d_frr_lo:+.3f}, {row.d_frr_hi:+.3f}] "
            f"({int(row.frr_a_better)}:{int(row.frr_b_better)})")


def write_summary(outputs: dict) -> None:
    """Compact markdown digest of the review analyses (every number is read from the CSVs written above)."""

    lines = ["# Analyses added after review", "",
             "Generated by `experiments/review_analyses.py` from `results/review/*.csv` "
             "(inputs: results/mswc, mswc_x, ssl, samespk_v2, noise_v2, detector_check, manuscript_checks).", ""]
    ab = outputs["ablation_mswc"]
    if not ab.empty:
        lines += ["## Paired FRR differences A - B, cross-speaker, target 1 FA/h", "",
                  "(95 % bootstrap over words; wakes where A is better : wakes where B is better)", ""]
        for cond in ("clean", "noisy"):
            for rule in ("oracle-wake", "bank60-tail"):
                sel = ab[(ab.condition == cond) & (ab.rule == rule) & (ab.target_fah == 1.0)]
                if sel.empty:
                    continue
                lines += [f"### {cond}, {rule}", "", "| contrast | A | B | k = 3 | k = 6 | k = 10 |", "|---|---|---|---|---|---|"]
                for contrast, g in sel.groupby("contrast", sort=False):
                    cells = {int(r.shots): fmt_pair(r) for r in g.itertuples()}
                    a, b = g.a.iloc[0], g.b.iloc[0]
                    lines.append(f"| {contrast} | {a} | {b} | " + " | ".join(cells.get(k, "-") for k in (3, 6, 10)) + " |")
                lines.append("")
    same = outputs["ablation_samespk"]
    if not same.empty:
        lines += ["## Same speaker (own clips masked from the seed-1 stream), paired FRR differences A - B, target 1", ""]
        for rule in ("oracle-wake", "bank60-tail"):
            sel = same[(same.rule == rule) & (same.target_fah == 1.0) & (same.condition == "clean")]
            lines += [f"### {rule}", "", "| set | k | contrast | FRR A | FRR B | dFRR | own-voice FA/h A | own-voice FA/h B |",
                      "|---|---|---|---|---|---|---|---|"]
            for r in sel.itertuples():
                lines.append(f"| {r.set} | {r.shots} | {r.contrast} | {r.frr_a:.3f} | {r.frr_b:.3f} | {fmt_pair(r)} | "
                             f"{r.fah_own_clean_a:.2f} | {r.fah_own_clean_b:.2f} |")
            lines.append("")
    own = outputs.get("samespk_own", pd.DataFrame())
    if not own.empty:
        lines += ["## Same speaker: own-voice alarms (target 1 FA/h)", "",
                  "| set | k | method | rule | wakes | FRR | FA/h other | own alarms / hours | own FA/h [speakers] [words] |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in own.itertuples():
            lines.append(f"| {r.set} | {r.shots} | {r.method} | {r.rule} | {r.wakes} | {r.frr:.3f} | {r.fah_other:.2f} | "
                         f"{r.own_alarms} / {r.own_hours:.2f} | {r.own_fah_pooled:.2f} [{r.own_fah_spk_lo:.2f}, "
                         f"{r.own_fah_spk_hi:.2f}] [{r.own_fah_word_lo:.2f}, {r.own_fah_word_hi:.2f}] |")
        lines.append("")
    pois = outputs.get("poisson_reference", pd.DataFrame())
    if not pois.empty:
        lines += ["## Poisson reference (cross-speaker, clean, target 1 FA/h)", "", "```",
                  pois.round(4).to_string(index=False), "```", ""]
    for name in ("rules_paired", "dev_split", "seed_spread"):
        t = outputs[name]
        lines += [f"## {name}", "", f"See `results/review/{name}.csv` ({len(t)} rows).", ""]
    tm = outputs["tail_models"]
    if not tm.empty:
        g = tm.groupby(["method", "model", "target_fah", "shots"]).agg(
            frr=("frr", "mean"), fah=("fah", "mean"), worst=("fah", "max"),
            within2=("fah", lambda x: float(np.mean(x <= 2.0 + 1e-9)))).reset_index()
        lines += ["## Tail models (cross-speaker, clean)", "", "| method | model | target | k | FRR | FA/h | worst | share <= 2 FA/h |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in g.itertuples():
            lines.append(f"| {r.method} | {r.model} | {r.target_fah} | {r.shots} | {r.frr:.3f} | {r.fah:.3f} | {r.worst:.2f} | {r.within2:.3f} |")
        ts = outputs["tail_shape"]
        lines += ["", "Slope of ln N(tau): median over wakes [IQR] of slope_eval / slope_bank:", ""]
        for method, gg in ts.groupby("method"):
            ratio = (gg.slope_eval / gg.slope_bank).dropna()
            lines.append(f"- {method}: {ratio.median():.3f} [{ratio.quantile(0.25):.3f}, {ratio.quantile(0.75):.3f}]; "
                         f"median slopes bank {gg.slope_bank.median():.1f}, eval (8-200 events / 4 h) "
                         f"{gg.slope_eval.median():.1f}, eval (2-50 events / 4 h) {gg.slope_eval_low.median():.1f}")
        lines.append("")
    for extra in ("noise_v2/summary.csv", "detector_check/summary.csv"):
        path = R / extra
        if path.is_file():
            t = pd.read_csv(path)
            lines += [f"## {extra}", "", "```", t.round(4).to_string(index=False), "```", ""]
    path = R / "manuscript_checks" / "sentence_overlap.json"
    if path.is_file():
        lines += ["## Sentence-overlap control", "", "```", path.read_text(), "```", ""]
    (OUT / "summary_review.md").write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    per = mswc_all()
    outputs = {}
    outputs["ablation_mswc"] = ablation_mswc(per, rng)
    outputs["ablation_samespk"] = ablation_samespk(rng)
    outputs["shots_new"] = shots_new(per[per.method.isin(["emb-only-noaug", "prototype-noaug", "emb-head",
                                                          "emb-head-aug", "ssl-dtw", "nexa-noaug", "nexa"])], rng)
    outputs["rules_paired"] = rules_paired(rng)
    outputs["dev_split"] = dev_split()
    outputs["seed_spread"] = seed_spread(per)
    tm, ts = tail_models()
    outputs["tail_models"] = tm
    outputs["tail_shape"] = ts
    outputs["enrollment"] = enrollment_counts()
    outputs["enroll_speaker_overlap"] = enroll_overlap()
    outputs["esc50_tail"] = esc50_tail()
    outputs["samespk_own"] = samespk_own(rng)
    outputs["samespk_words"] = samespk_words()
    outputs["poisson_reference"] = poisson_reference(rng)
    outputs["samespk_own_curve"] = samespk_own_curve()
    for name, table in outputs.items():
        table.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}: {len(table)} rows")
    write_summary(outputs)


if __name__ == "__main__":
    main()
