"""Build results/summary.md and the compact summary CSVs from the calibration tables.

Usage: python experiments/make_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

R = common.RESULTS
NAMES = {
    "nexa": "deployed matcher (noise-augmented templates)",
    "nexa-min": "deployed, best single template",
    "nexa-noaug": "clean templates only",
    "nexa-noaug-bnorm": "clean templates, bank normaliser",
    "nexa-nonorm": "deployed, no normaliser",
    "emb-only": "embedding part only (96-d)",
    "mel-only": "log-mel part only (32-d)",
    "prototype": "prototype (mean embedding)",
    "mfcc-dtw": "MFCC + DTW",
    "nexa-noaug-cn": "clean templates, condition-matched normaliser",
    "nexa-cn": "noise-augmented templates, condition-matched normaliser",
}
RULES = {
    "deployed": "deployed (20 s room audio)",
    "global-bank": "one global threshold (bank, other wakes)",
    "bank15": "bank 15 min, empirical",
    "bank30": "bank 30 min, empirical",
    "bank60": "bank 60 min, empirical",
    "bank60-os": "bank 60 min, order statistic",
    "bank60-tail": "bank 60 min, tail fit",
    "bank30-tail": "bank 30 min, tail fit",
    "bank15-tail": "bank 15 min, tail fit",
    "bank60+noisy": "bank 60 min + noisy copy, empirical",
    "oracle-global": "oracle global (eval negatives)",
    "oracle-wake": "oracle per wake (eval negatives)",
    "speechbank": "speech bank 60 min, empirical",
    "speechbank-os": "speech bank, order statistic",
    "speechbank-tail": "speech bank, tail fit",
    "speech+soundbank": "speech + sound bank, empirical",
    "speech+soundbank-tail": "speech + sound bank, tail fit",
}


def fmt_ci(row, col="frr") -> str:
    return f"{row[col]:.3f} [{row[col + '_lo']:.3f}, {row[col + '_hi']:.3f}]"


def table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(c[1] for c in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    lines = [head, sep]
    for _, row in frame.iterrows():
        cells = []
        for key, _label in columns:
            value = row[key] if not callable(key) else key(row)
            if isinstance(value, float):
                value = f"{value:.3f}" if abs(value) < 100 else f"{value:.1f}"
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def enrollment_rejections() -> dict[int, int]:
    counts = {3: 0, 6: 0, 10: 0}
    for path in (R / "mswc" / "tasks").glob("*.json"):
        r = json.loads(path.read_text())
        counts[r["shots"]] += int(r["enrolled"] < r["shots"])
    return counts


def load(name: str) -> pd.DataFrame:
    path = R / name / "calibration.csv"
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _row(frame: pd.DataFrame, **match) -> pd.Series:
    sel = frame
    for key, value in match.items():
        sel = sel[sel[key] == value]
    if len(sel) != 1:
        raise KeyError(f"{len(sel)} rows for {match}")
    return sel.iloc[0]


def _span(values, fmt="{:.2f}") -> str:
    lo, hi = min(values), max(values)
    return fmt.format(lo) if fmt.format(lo) == fmt.format(hi) else f"{fmt.format(lo)} to {fmt.format(hi)}"


def findings(mswc: pd.DataFrame, same: pd.DataFrame) -> list[str]:
    """Main findings, every number read from the calibration and paired tables."""

    ks = (3, 6, 10)
    clean = mswc[(mswc.condition == "clean")]
    get = lambda method, rule, k, target=1.0, frame=clean: _row(frame, method=method, rule=rule, shots=k,  # noqa: E731
                                                                 target_fah=target)
    out = ["## 0. Main findings (numbers from the tables below)\n"]
    dep = [get("nexa", "deployed", k) for k in ks]
    dep_clean = [get("nexa-noaug", "deployed", k) for k in ks]
    out.append(f"1. **The deployed calibration (20 s of room audio) is far too loose.** Cross-speaker, clean: the "
               f"deployed matcher reaches a mean of {_span([r.fah_mean for r in dep], '{:.0f}')} FA/h over k = 3, 6, "
               f"10 (median {_span([r.fah_median for r in dep], '{:.0f}')}), the clean-template variant "
               f"{_span([r.fah_mean for r in dep_clean], '{:.0f}')} FA/h; at most "
               f"{max(r.share_within_2x for r in dep + dep_clean):.0%} of the 48 wakes stay within 2 FA/h.")
    tail = {t: [get("nexa-noaug", "bank60-tail", k, t) for k in ks] for t in (0.5, 1.0, 2.0)}
    t1 = tail[1.0]
    nexa_tail = [get("nexa", "bank60-tail", k) for k in ks]
    out.append(f"2. **Per-wake calibration on the 60-min bank (11.5 MB at fp16) with a tail fit lands near the "
               f"target on disjoint negatives.** Clean templates, target 1 FA/h: mean "
               f"{_span([r.fah_mean for r in t1])} FA/h over k (median {_span([r.fah_median for r in t1])}, worst wake "
               f"{_span([r.fah_max for r in t1])}), {_span([r.share_within_2x for r in t1], '{:.0%}')} of wakes within "
               f"2 FA/h, at FRR {_span([r.frr for r in t1], '{:.3f}')}. Targets 0.5 and 2 FA/h give "
               f"{_span([r.fah_mean for r in tail[0.5]])} and {_span([r.fah_mean for r in tail[2.0]])} FA/h. "
               f"The deployed matcher with the same rule: {_span([r.fah_mean for r in nexa_tail])} FA/h at target 1.")
    emp = [get("nexa-noaug", "bank60", k) for k in ks]
    os_ = [get("nexa-noaug", "bank60-os", k) for k in ks]
    out.append(f"3. **The naive empirical rule overshoots by design.** Taking the largest threshold whose bank rate "
               f"is still <= 1 FA/h allows one bank event in 1 h and gives {_span([r.fah_mean for r in emp])} FA/h; "
               f"the order-statistic rule (no bank event allowed) gives {_span([r.fah_mean for r in os_])} FA/h but cannot "
               f"distinguish targets below one event per bank length (0.5 and 1 FA/h give the same threshold); the "
               f"tail fit can.")
    glob6 = get("nexa-noaug", "global-bank", 6)
    tail6 = get("nexa-noaug", "bank60-tail", 6)
    globn = get("nexa", "global-bank", 6)
    tailn = get("nexa", "bank60-tail", 6)
    paired = R / "mswc" / "paired.csv"
    text = ""
    if paired.is_file():
        p = pd.read_csv(paired)
        pr = p[(p.condition == "clean") & (p.target_fah == 1.0) & (p.method == "nexa-noaug") & (p.shots == 6)
               & (p.rule_a == "bank60-tail") & (p.rule_b == "global-bank")]
        if len(pr) == 1:
            pr = pr.iloc[0]
            text = (f" Paired over the 48 wakes, per-wake minus global: dFRR {pr.d_frr:+.3f} "
                    f"[{pr.d_frr_lo:+.3f}, {pr.d_frr_hi:+.3f}], dFA/h {pr.d_fah:+.2f} [{pr.d_fah_lo:+.2f}, "
                    f"{pr.d_fah_hi:+.2f}]"
                    + ("; neither interval excludes 0." if pr.d_frr_lo < 0 < pr.d_frr_hi and pr.d_fah_lo < 0 < pr.d_fah_hi
                       else "."))
    out.append(f"4. **One global threshold learned on the bank vs. per-wake bank calibration.** Clean templates, k = 6, target 1: global FRR {glob6.frr:.3f}, mean "
               f"{glob6.fah_mean:.2f} FA/h, worst wake {glob6.fah_max:.2f} FA/h; per-wake tail fit FRR {tail6.frr:.3f}, "
               f"mean {tail6.fah_mean:.2f}, worst wake {tail6.fah_max:.2f}. For the deployed matcher the global "
               f"threshold's worst wake reaches {globn.fah_max:.2f} FA/h (per-wake: {tailn.fah_max:.2f}).{text}")
    pm = R / "mswc" / "paired_methods.csv"
    if pm.is_file():
        pm = pd.read_csv(pm)
        pm = pm[(pm.condition == "clean") & (pm.target_fah == 1.0) & (pm.rule == "oracle-wake")]
        pair = lambda a, b, k: _row(pm, a=a, b=b, shots=k)  # noqa: E731
        aug = [pair("nexa-noaug", "nexa", k) for k in ks]
        norm = [pair("nexa", "nexa-nonorm", k) for k in ks]
        bnorm = [pair("nexa-noaug-bnorm", "nexa-noaug", k) for k in ks]
        mf = [pair("nexa-noaug", "mfcc-dtw", k) for k in ks]
        emb = [pair("nexa-noaug", "emb-only", k) for k in ks]
        proto = [pair("nexa-noaug", "prototype", k) for k in ks]
        mel = [pair("nexa-noaug", "mel-only", k) for k in ks]
        nonorm = [get("nexa-nonorm", "oracle-wake", k) for k in ks]
        out.append(f"5. **Method comparison at matched FA/h (each wake's own 1 FA/h threshold on the evaluation "
                   f"negatives), cross-speaker.** Dropping the noise-augmented template copies lowers FRR from "
                   f"{_span([r.frr_b for r in aug], '{:.3f}')} to {_span([r.frr_a for r in aug], '{:.3f}')} "
                   f"(paired dFRR {_span([r.d_frr for r in aug], '{:+.3f}')}; "
                   f"{sum(r.d_frr_hi < 0 for r in aug)} of 3 intervals entirely below 0). Without "
                   f"the background normaliser FRR is {_span([r.frr for r in nonorm], '{:.3f}')} (deployed minus "
                   f"no-normaliser {_span([r.d_frr for r in norm], '{:+.3f}')}). Fitting the normaliser on the bank "
                   f"instead of the room audio changes FRR by {_span([r.d_frr for r in bnorm], '{:+.3f}')}. Clean "
                   f"templates beat the 96-d embedding alone by {_span([-r.d_frr for r in emb], '{:.3f}')}, MFCC + DTW "
                   f"by {_span([-r.d_frr for r in mf], '{:.3f}')}, the mean-embedding prototype by "
                   f"{_span([-r.d_frr for r in proto], '{:.3f}')} and the 32-d log-mel part alone by "
                   f"{_span([-r.d_frr for r in mel], '{:.3f}')} in FRR. Even the best variant misses "
                   f"{_span([get('nexa-noaug-bnorm', 'oracle-wake', k).frr for k in ks], '{:.0%}')} of cross-speaker "
                   f"positives at 1 FA/h.")
    pk = R / "mswc" / "paired_k.csv"
    if pk.is_file():
        pk = pd.read_csv(pk)
        pk = pk[(pk.condition == "clean") & (pk.target_fah == 1.0) & (pk.rule == "oracle-wake")]
        a = _row(pk, method="nexa-noaug", a=10, b=3)
        b = _row(pk, method="nexa", a=10, b=3)
        c = _row(pk, method="nexa-noaug", a=10, b=6)
        out.append(f"6. **Effect of the number of enrollment shots.** "
                   f"k = 10 minus k = 3 at matched FA/h: clean templates dFRR {a.d_frr:+.3f} [{a.d_frr_lo:+.3f}, "
                   f"{a.d_frr_hi:+.3f}] (k = 10 minus 6: {c.d_frr:+.3f} [{c.d_frr_lo:+.3f}, {c.d_frr_hi:+.3f}]); "
                   f"deployed matcher {b.d_frr:+.3f} [{b.d_frr_lo:+.3f}, {b.d_frr_hi:+.3f}].")
    noisy = mswc[mswc.condition == "noisy"]
    nn = _row(noisy, method="nexa", rule="bank60-tail", shots=6, target_fah=1.0)
    nc = _row(noisy, method="nexa-noaug", rule="bank60-tail", shots=6, target_fah=1.0)
    nd = _row(noisy, method="nexa", rule="deployed", shots=6, target_fah=1.0)
    out.append(f"7. **Thresholds set on clean audio do not transfer to 10 dB noise that appears after enrollment.** "
               f"k = 6: the deployed matcher (noise-augmented templates) gives {nd.fah_mean:.0f} FA/h with the "
               f"deployed calibration and {nn.fah_mean:.0f} FA/h with the bank tail fit; clean templates stay "
               f"quiet ({nc.fah_mean:.2f} FA/h) but miss {nc.frr:.0%} of positives.")
    if not same.empty:
        sc = same[(same.condition == "clean") & (same.target_fah == 1.0)]
        g = lambda method, rule, setname="main", k=10: _row(sc, set=setname, method=method, rule=rule,  # noqa: E731
                                                             shots=k)
        st = g("nexa-noaug", "bank60-tail")
        sd = g("nexa-noaug", "deployed")
        sm = g("mfcc-dtw", "bank60-tail")
        sn = g("nexa", "bank60-tail")
        sb = g("nexa-noaug", "bank60-tail", "broad", 3)
        out.append(f"8. **Same-speaker enrollment (MSWC speakers who repeat a word, main set, k = 10, "
                   f"{int(st.wakes)} speakers).** Clean templates with the bank tail fit: FRR {st.frr:.3f} "
                   f"[{st.frr_lo:.3f}, {st.frr_hi:.3f}] at {st.fah_mean:.2f} FA/h on other speakers but "
                   f"{st.fah_own_clean_pooled:.2f} FA/h on the same speaker's other words "
                   f"({st.hours_own_clean_total:.1f} h pooled); with the deployed calibration the same templates give FRR "
                   f"{sd.frr:.3f} at {sd.fah_mean:.0f} FA/h. The deployed matcher with the bank tail fit: FRR "
                   f"{sn.frr:.3f}. MFCC + DTW reaches FRR {sm.frr:.3f} at {sm.fah_mean:.2f} FA/h on other speakers but "
                   f"{sm.fah_own_clean_pooled:.1f} FA/h on the speaker's own other words: it responds to the voice, not "
                   f"only to the word. Broad set (k = 3, {int(sb.wakes)} speakers): clean templates FRR "
                   f"{sb.frr:.3f} at {sb.fah_mean:.2f} FA/h (own voice {sb.fah_own_clean_pooled:.2f}).")
    out += extra_findings()
    out.append("")
    return out


def extra_findings() -> list[str]:
    """Findings 9-12: out-of-domain speech, noise diagnostics and the non-speech wakes (part 2)."""

    out = []
    fl = load("fleurs")
    if not fl.empty:
        f = fl[(fl.condition == "clean") & (fl.target_fah == 1.0)]
        t = _row(f, method="nexa-noaug", rule="bank60-tail")
        d = _row(f, method="nexa", rule="deployed")
        dn = _row(f, method="nexa-noaug", rule="deployed")
        info = common.load_json(R / "fleurs" / "streams" / "info.json")
        out.append(f"9. **The bank thresholds stay conservative on out-of-domain continuous speech.** On "
                   f"{info['seconds'] / 3600:.2f} h of FLEURS Persian read speech the 48 cross-speaker wakes (k = 6) "
                   f"with clean templates and the bank tail fit give {t.fah_mean:.3f} FA/h (worst wake "
                   f"{t.fah_max:.2f}); the deployed calibration gives {d.fah_mean:.1f} FA/h (deployed matcher) and "
                   f"{dn.fah_mean:.1f} FA/h (clean templates). Isolated words are the harder negative material.")
    noise = R / "mswc" / "noise_only.csv"
    cn = load("condnorm")
    if noise.is_file():
        n = pd.read_csv(noise)
        g = n.groupby("method").noise_only_fah.agg(["mean", "median", "max"])
        text = (f"10. **The noise-augmented templates match noise itself.** On {n.noise_hours.iloc[0] * 60:.0f} min "
                f"of pure white + pink noise (seed 1, k = 6, clean-bank threshold at 1 FA/h) the deployed matcher "
                f"fires {g.loc['nexa', 'mean']:.0f} times per hour on average (median {g.loc['nexa', 'median']:.0f}, "
                f"max {g.loc['nexa', 'max']:.0f}); clean templates fire {g.loc['nexa-noaug', 'mean']:.0f} times.")
        if not cn.empty:
            c = cn[(cn.target_fah == 1.0) & (cn.rule == "bank60-tail") & (cn.shots == 6)]
            a = _row(c, method="nexa-cn", condition="noisy")
            b = _row(c, method="nexa-noaug-cn", condition="noisy")
            o = _row(cn[(cn.target_fah == 1.0) & (cn.shots == 6)], method="nexa-noaug-cn", condition="noisy",
                     rule="oracle-wake")
            text += (f" Normalising every stream with the statistics of its own condition (exploratory, seed 1) "
                     f"removes the false-alarm burst in noise ({a.fah_mean:.2f} and {b.fah_mean:.2f} FA/h with the "
                     f"bank tail fit) but not the misses (FRR {a.frr:.3f} and {b.frr:.3f}; {o.frr:.3f} even with each "
                     f"wake's own threshold on the noisy negatives).")
        out.append(text)
    esc = load("esc50")
    if not esc.empty:
        e = esc[(esc.target_fah == 1.0) & (esc.shots == 6)]
        dep = _row(e, method="nexa", rule="deployed")
        sp = _row(e, method="nexa-noaug", rule="speechbank-tail")
        sb = _row(e, method="nexa-noaug", rule="speech+soundbank")
        orc = e[e.rule == "oracle-wake"].sort_values("frr")
        best = orc.iloc[0]
        e10 = esc[(esc.target_fah == 10.0) & (esc.shots == 6) & (esc.rule == "oracle-wake")].sort_values("frr")
        text = (f"11. **Few-shot sound wakes (ESC-50, k = 6, 150 wakes from 50 classes, cross-recording) are not "
                f"usable at 1 FA/h against dense environmental sound.** The deployed calibration gives "
                f"{dep.fah_mean:.0f} FA/h (FRR {dep.frr:.3f}); a speech-only bank cannot calibrate a sound wake "
                f"(clean templates, tail fit: {sp.fah_mean:.0f} FA/h over speech and sounds together; "
                f"{sp.fah_speech_eval_mean:.1f} FA/h on the speech stream, {sp.fah_sound_eval_mean:.0f} FA/h on the "
                f"sound stream). Adding "
                f"{common.load_json(R / 'esc50' / 'streams' / 'info.json')['bank']['seconds'] / 3600:.2f} h of ESC-50 "
                f"folds 1-2 (other classes) to the bank meets the target ({sb.fah_mean:.2f} FA/h) but FRR is "
                f"{sb.frr:.3f}; even each "
                f"wake's own threshold on the evaluation negatives leaves FRR >= {best.frr:.3f} (best method: "
                f"{NAMES[best.method]}).")
        if not e10.empty:
            text += (f" At 10 FA/h (own threshold) the best FRR is {e10.iloc[0].frr:.3f} "
                     f"({NAMES[e10.iloc[0].method]}).")
        out.append(text)
    ss = load("esc50_samesrc")
    xc_path = R / "esc50_xclass" / "paired.csv"
    if not ss.empty or xc_path.is_file():
        text = "12. **Same-source enrollment and unseen sound types.**"
        if not ss.empty:
            s = ss[(ss.target_fah == 1.0)]
            a = _row(s, method="nexa-noaug", rule="speech+soundbank")
            orc = s[s.rule == "oracle-wake"].sort_values("frr").iloc[0]
            text += (f" Enrolling 3 takes of the same recording and testing on its other takes "
                     f"({int(a.wakes)} wakes): clean templates with the speech + sound bank reach FRR {a.frr:.3f} "
                     f"[{a.frr_lo:.3f}, {a.frr_hi:.3f}] at {a.fah_mean:.2f} FA/h; the best own-threshold FRR is "
                     f"{orc.frr:.3f} ({NAMES[orc.method]}).")
        if xc_path.is_file():
            xp = pd.read_csv(xc_path)
            xc = pd.read_csv(R / "esc50_xclass" / "calibration.csv")
            r = _row(xp, method="nexa-noaug", shots=6, target_fah=1.0)
            seen = _row(xc, method="nexa-noaug", shots=6, target_fah=1.0, condition="seen")
            unseen = _row(xc, method="nexa-noaug", shots=6, target_fah=1.0, condition="unseen")
            text += (f" A sound bank that lacks the evaluation sound types (class-disjoint halves, clean templates, "
                     f"k = 6) raises the sound-stream FA/h from {seen.fah_sound_mean:.2f} to "
                     f"{unseen.fah_sound_mean:.2f} (paired {r.d_fah_sound:+.2f} [{r.d_fah_sound_lo:+.2f}, "
                     f"{r.d_fah_sound_hi:+.2f}]) at identical FRR.")
        out.append(text)
    out += cost_finding()
    return out


def _cost_tables():
    """The compute-cost tables written by experiments/cost.py, or None if it has not been run."""

    base = R / "cost"
    if not (base / "matcher_cost.csv").is_file():
        return None
    return (common.load_json(base / "cost_info.json"), pd.read_csv(base / "matcher_cost.csv"),
            pd.read_csv(base / "mfcc_dtw_cost.csv"), pd.read_csv(base / "bank_calibration_cost.csv"))


def cost_finding() -> list[str]:
    tables = _cost_tables()
    if tables is None:
        return []
    info, c, md, b = tables
    feat = info["features"]["feature_ms_median"]
    typical = info["template_length_median"]
    a = c[c.length == typical].set_index("templates")
    md = md.set_index("templates")
    b = b.set_index("templates")
    text = (f"13. **Compute cost is small next to the 80 ms frame (single thread, x86 virtual machine; Raspberry Pi "
            f"not measured).** Feature extraction takes {feat:.2f} ms per frame. With {typical}-frame templates the "
            f"deployed NumPy matcher and decision layer add {a.loc[6, 'detector_ms_median']:.2f} ms for 6 clean "
            f"templates and {a.loc[18, 'detector_ms_median']:.2f} ms for 18 (k = 6 with the two noise copies), "
            f"{a.loc[240, 'detector_ms_median']:.2f} ms for 240; the compiled matcher needs "
            f"{a.loc[18, 'compiled_ms']:.3f} ms for 18. Features + matcher for one k = 6 wake use "
            f"{100 * a.loc[18, 'real_time_factor']:.1f} % of real time. Calibrating one wake on the 60-min bank "
            f"takes {b.loc[6, 'compiled_seconds']:.2f} s (6 templates, compiled matcher)")
    if "deployed_seconds" in b and np.isfinite(b.loc[6, "deployed_seconds"]):
        text += f" or {b.loc[6, 'deployed_seconds']:.1f} s with the deployed NumPy code"
    text += "."
    if 6 in md.index and 6 in a.index:
        mf = md.loc[6]
        text += (f" The MFCC + DTW baseline (6 templates of {int(mf.length)} 10-ms frames) costs "
                 f"{mf.total_ms_per_80ms:.2f} ms per 80 ms of audio (compiled DTW; batch MFCC "
                 f"{info['mfcc_features']['mfcc_ms_per_80ms']:.2f} ms), "
                 f"against {a.loc[6, 'total_ms_compiled']:.2f} ms for features + compiled matcher with 6 clean "
                 f"templates.")
    return [text]


def main() -> None:
    out = []
    split = common.load_json(R / "splits.json")
    stats = split["stats"]
    bank = common.load_json(R / "bank" / "bank_info.json")
    streams = {s: common.load_json(R / "mswc" / "streams" / f"seed{s}" / "info.json") for s in (1, 2, 3)}
    out.append("# Few-shot wake detection: results of the step-1 experiments\n")
    out.append("All numbers below were produced by the scripts in `experiments/` on this machine; raw "
               "per-task results are in `results/*/tasks/*.json`, per-wake rows in `results/*/per_task.csv`, "
               "aggregates with 95 % cluster-bootstrap intervals (resampling words, speakers or classes) in "
               "`results/*/calibration.csv`.\n")
    out.append("## Data and partition\n")
    out.append(f"- Persian MSWC test split: {stats['clips_total']:,} one-second clips, {stats['speakers_total']:,} "
               f"speakers, {stats['words_total']:,} words.")
    overlap = len(set(common.TARGETS) & set(split["same_words"]))
    out.append(f"- Evaluation words never used as negatives or in the bank: {stats['target_set']} distinct words "
               f"(the 16 cross-speaker targets and the {stats['same_words']} words that some speaker repeats >= 8 "
               f"times; {overlap} words are on both lists).")
    out.append(f"- Bank: {bank['clips']:,} clips ({bank['seconds'] / 60:.1f} min with 0.25 s gaps) from "
               f"{stats['bank_unique_speakers']} bank-only speakers and {stats['bank_unique_words']} bank-only words; "
               f"{bank['clean_frames']:,} feature frames = {bank['clean_feat_MB_fp16']} MB at fp16 "
               f"(noisy copy another {bank['noisy_feat_MB_fp16']} MB; MFCC copy for the baseline "
               f"{bank['clean_mfcc_MB_fp16']} MB).")
    out.append(f"- Evaluation negatives per seed: {streams[1]['negative_clips']:,} clips = "
               + ", ".join(f"seed {s}: {v['clean_seconds'] / 3600:.3f} h" for s, v in streams.items())
               + f" (clean; the same audio with white+pink noise at 10 dB SNR as the noisy condition). They share "
               f"no speaker and no word with the bank and contain none of the {stats['target_set']} evaluation words; "
               f"the three seeds use different clips (speakers and words do recur across seeds). "
               f"Room audio for the deployed calibration: {streams[1]['room_seconds']:.1f} s per seed.")
    total_h = sum(v["clean_seconds"] for v in streams.values()) / 3600
    out.append(f"- FA/h resolution: one false alarm on a 4 h stream is 0.25 FA/h for that wake; per k the 48 "
               f"cross-speaker wakes are scored on {48 * total_h / 3:.0f} wake-hours of clean negatives "
               f"({total_h:.1f} h of distinct audio over the three seeds).")
    contains = [sum(any(t in w for t in common.TARGETS) for w in split["seeds"][str(s)]["negative_words"])
                for s in (1, 2, 3)]
    out.append(f"- Near-miss words: the evaluation negatives exclude the target words themselves, but "
               f"{min(contains)}-{max(contains)} of the 11,520 clips per seed are other words that contain a target "
               f"as a substring (e.g. دوستان for دوست); they are kept as negatives.")
    out.append("- Cross-speaker positives: 60 recordings by 60 speakers not enrolled, per word and seed "
               "(16 words x 3 seeds = 48 wakes per k, 2,880 positives per k). Enrollment uses k recordings by k "
               "different speakers; nested across k (k = 3 is a subset of k = 6, which is a subset of k = 10). "
               "The deployed enrollment rejected a recording in "
               + ", ".join(f"{n} of 48 wakes at k = {k}" for k, n in enrollment_rejections().items())
               + " (the wake was still built from the remaining recordings).\n")

    mswc = load("mswc")
    if not mswc.empty:
        out.extend(findings(mswc, load("samespk")))
        out.append("## 1. Calibration against the shipped negative bank (cross-speaker, clean)\n")
        for k in (6,):
            sel = mswc[(mswc.shots == k) & (mswc.condition == "clean") & (mswc.target_fah == 1.0)]
            for method in ("nexa", "nexa-noaug"):
                part = sel[(sel.method == method) & sel.rule.isin(
                    ["deployed", "global-bank", "bank15", "bank30", "bank60", "bank60-os", "bank30-tail",
                     "bank60-tail", "oracle-global", "oracle-wake"])].copy()
                part["rule_name"] = part.rule.map(RULES)
                part["frr_ci"] = part.apply(fmt_ci, axis=1)
                part["fah_ci"] = part.apply(lambda r: f"{r.fah_mean:.2f} [{r.fah_lo:.2f}, {r.fah_hi:.2f}]", axis=1)
                order = ["deployed", "global-bank", "bank15", "bank30", "bank60", "bank60-os", "bank30-tail",
                         "bank60-tail", "oracle-global", "oracle-wake"]
                part = part.set_index("rule").loc[[o for o in order if o in set(part.rule)]].reset_index()
                out.append(f"**{NAMES[method]}, k = {k}, target 1 FA/h, {int(part.wakes.iloc[0])} wakes**\n")
                out.append(table(part, [("rule_name", "threshold rule"), ("frr_ci", "FRR [95% CI]"),
                                        ("fah_ci", "FA/h mean [95% CI]"), ("fah_median", "FA/h median"),
                                        ("fah_p90", "FA/h p90"), ("fah_max", "FA/h worst wake"),
                                        ("share_within_2x", "share of wakes <= 2 FA/h")]))
                out.append("")
        sel = mswc[(mswc.condition == "clean") & (mswc.method == "nexa-noaug") & mswc.rule.isin(
            ["bank60", "bank60-os", "bank60-tail", "global-bank", "oracle-wake"])]
        piv = sel.pivot_table(index=["rule"], columns=["target_fah", "shots"], values="fah_mean")
        out.append("**Achieved FA/h on the evaluation negatives vs. target (clean templates; columns = target, k)**\n")
        piv.columns = [f"{t:g} FA/h, k={k}" for t, k in piv.columns]
        piv = piv.reset_index()
        piv["rule"] = piv.rule.map(RULES)
        out.append(table(piv, [(c, c) for c in piv.columns]))
        out.append("")
        piv = sel.pivot_table(index=["rule"], columns=["target_fah", "shots"], values="frr")
        out.append("**FRR at the same thresholds (clean templates; columns = target, k)**\n")
        piv.columns = [f"{t:g} FA/h, k={k}" for t, k in piv.columns]
        piv = piv.reset_index()
        piv["rule"] = piv.rule.map(RULES)
        out.append(table(piv, [(c, c) for c in piv.columns]))
        out.append("")
        sel1 = mswc[(mswc.condition == "clean") & (mswc.target_fah == 1.0) & mswc.method.isin(["nexa", "nexa-noaug"])
                    & mswc.rule.isin(["global-bank", "oracle-global", "bank60", "bank60-os", "bank60-tail"])].copy()
        sel1["method"] = sel1.method.map(NAMES)
        sel1["rule"] = sel1.rule.map(RULES)
        sel1 = sel1.sort_values(["method", "rule", "shots"])
        out.append("**Spread of the achieved FA/h over wakes (cross-speaker, clean, target 1 FA/h; 48 wakes per row)**\n")
        out.append(table(sel1, [("method", "method"), ("rule", "rule"), ("shots", "k"), ("frr", "FRR"),
                                ("fah_mean", "FA/h mean"), ("fah_median", "median"), ("fah_p90", "p90"),
                                ("fah_max", "worst wake"), ("share_within_2x", "share <= 2 FA/h")]))
        out.append("")
        tails = [r for r in sorted(set(mswc.rule)) if r.startswith("bank60-tail")]
        part = mswc[(mswc.shots == 6) & (mswc.condition == "clean") & (mswc.method == "nexa-noaug")
                    & (mswc.target_fah == 1.0) & mswc.rule.isin(tails)].copy()
        if not part.empty:
            part["fit range (bank events)"] = part.rule.map(lambda r: "[2, 50] (default)" if r == "bank60-tail"
                                                            else r.replace("bank60-tail", ""))
            out.append("**Tail-fit sensitivity to the fitted event range (clean templates, k = 6, target 1 FA/h)**\n")
            out.append(table(part, [("fit range (bank events)", "fit range (bank events)"), ("frr", "FRR"),
                                    ("fah_mean", "FA/h mean"), ("fah_median", "FA/h median"),
                                    ("share_within_2x", "share of wakes <= 2 FA/h")]))
            out.append("")
        paired = R / "mswc" / "paired.csv"
        if paired.is_file():
            p = pd.read_csv(paired)
            p = p[(p.condition == "clean") & (p.target_fah == 1.0) & p.method.isin(["nexa", "nexa-noaug"])
                  & (p.rule_a == "bank60-tail") & p.rule_b.isin(["deployed", "global-bank", "oracle-wake", "bank60"])]
            p = p.sort_values(["method", "rule_b", "shots"]).copy()
            p["method"] = p.method.map(NAMES)
            p["compared with"] = p.rule_b.map(RULES)
            p["dFRR [95% CI]"] = p.apply(lambda r: f"{r.d_frr:+.3f} [{r.d_frr_lo:+.3f}, {r.d_frr_hi:+.3f}]", axis=1)
            p["dFA/h [95% CI]"] = p.apply(lambda r: f"{r.d_fah:+.2f} [{r.d_fah_lo:+.2f}, {r.d_fah_hi:+.2f}]", axis=1)
            out.append("**Paired differences, bank 60 min tail fit minus another rule (same wakes; cross-speaker, "
                       "clean, target 1 FA/h; 95 % bootstrap over words)**\n")
            out.append(table(p, [("method", "method"), ("shots", "k"), ("compared with", "compared with"),
                                 ("wakes", "wakes"), ("dFRR [95% CI]", "dFRR [95% CI]"),
                                 ("dFA/h [95% CI]", "dFA/h [95% CI]")]))
            out.append("")

        out.append("## 2. Methods and enrollment size (cross-speaker, clean, target 1 FA/h)\n")
        out.append("FRR (achieved FA/h) at the deployable bank 60 min tail-fit threshold; FRR at the oracle global "
                   "threshold and at each wake's own oracle threshold (eval negatives) as references. 48 wakes per cell.\n")
        rows = []
        for method in [m for m in NAMES if m in set(mswc.method)]:
            row = {"method": NAMES[method]}
            for k in (3, 6, 10):
                a = mswc[(mswc.method == method) & (mswc.shots == k) & (mswc.condition == "clean")
                         & (mswc.target_fah == 1.0)]
                t = a[a.rule == "bank60-tail"]
                o = a[a.rule == "oracle-global"]
                w = a[a.rule == "oracle-wake"]
                if not t.empty:
                    row[f"k={k} FRR (FA/h)"] = f"{t.frr.iloc[0]:.3f} ({t.fah_mean.iloc[0]:.2f})"
                    row[f"k={k} oracle-global FRR"] = f"{o.frr.iloc[0]:.3f}"
                    row[f"k={k} oracle/wake FRR"] = f"{w.frr.iloc[0]:.3f}"
            rows.append(row)
        frame = pd.DataFrame(rows)
        out.append(table(frame, [(c, c) for c in frame.columns]))
        out.append("")
        out.append("Deployed calibration (20 s room audio) for comparison:\n")
        rows = []
        for method in ("nexa", "nexa-noaug", "nexa-nonorm"):
            row = {"method": NAMES[method]}
            for k in (3, 6, 10):
                a = mswc[(mswc.method == method) & (mswc.shots == k) & (mswc.condition == "clean")
                         & (mswc.target_fah == 1.0) & (mswc.rule == "deployed")]
                if not a.empty:
                    row[f"k={k} FRR / FA/h"] = f"{a.frr.iloc[0]:.3f} / {a.fah_mean.iloc[0]:.1f}"
            rows.append(row)
        frame = pd.DataFrame(rows)
        out.append(table(frame, [(c, c) for c in frame.columns]))
        out.append("")

        pm_path = R / "mswc" / "paired_methods.csv"
        if pm_path.is_file():
            pm = pd.read_csv(pm_path)
            pm = pm[(pm.condition == "clean") & (pm.target_fah == 1.0) & pm.rule.isin(["oracle-wake", "bank60-tail"])]
            rows = []
            for (a, b), frame in pm.groupby(["a", "b"], sort=False):
                row = {"method A": NAMES[a], "method B": NAMES[b]}
                for rule in ("oracle-wake", "bank60-tail"):
                    for k in (3, 6, 10):
                        r = frame[(frame.rule == rule) & (frame.shots == k)]
                        if r.empty:
                            continue
                        r = r.iloc[0]
                        short = {"oracle-wake": "oracle/wake", "bank60-tail": "bank tail"}[rule]
                        row[f"{short} k={k}"] = (f"{r.d_frr:+.3f} [{r.d_frr_lo:+.3f}, {r.d_frr_hi:+.3f}] "
                                                 f"({int(r.frr_a_better)}:{int(r.frr_b_better)})")
                rows.append(row)
            frame = pd.DataFrame(rows)
            out.append("**Paired method differences, FRR(A) - FRR(B) on the same 48 wakes per k (cross-speaker, clean, "
                       "target 1 FA/h; 95 % bootstrap over words; in brackets after the interval: wakes where A has the "
                       "lower FRR : wakes where B has). 'oracle/wake' = each wake's threshold set to 1 FA/h on its own "
                       "evaluation negatives, so both methods run at about the same FA/h and the difference is a "
                       "difference of DET operating points; 'bank tail' = the deployable bank rule, where the two "
                       "methods may also differ in achieved FA/h (see the table above).**\n")
            out.append(table(frame, [(c, c) for c in frame.columns]))
            out.append("")
        pk_path = R / "mswc" / "paired_k.csv"
        if pk_path.is_file():
            pk = pd.read_csv(pk_path)
            pk = pk[(pk.condition == "clean") & (pk.target_fah == 1.0) & pk.rule.isin(["oracle-wake", "bank60-tail"])]
            rows = []
            for method in [m for m in NAMES if m in set(pk.method)]:
                row = {"method": NAMES[method]}
                for rule in ("oracle-wake", "bank60-tail"):
                    for a, b in ((6, 3), (10, 6), (10, 3)):
                        r = pk[(pk.method == method) & (pk.rule == rule) & (pk.a == a) & (pk.b == b)]
                        if r.empty:
                            continue
                        r = r.iloc[0]
                        short = {"oracle-wake": "oracle/wake", "bank60-tail": "bank tail"}[rule]
                        row[f"{short} k{a}-k{b}"] = f"{r.d_frr:+.3f} [{r.d_frr_lo:+.3f}, {r.d_frr_hi:+.3f}]"
                rows.append(row)
            frame = pd.DataFrame(rows)
            out.append("**Effect of the number of enrollment shots, FRR(k = a) - FRR(k = b) on the same 48 wakes "
                       "(nested enrollment, same positives and negatives; cross-speaker, clean, target 1 FA/h; 95 % "
                       "bootstrap over words)**\n")
            out.append(table(frame, [(c, c) for c in frame.columns]))
            out.append("")

        out.append("## 3. Noisy condition (white + pink noise at 10 dB SNR, appearing after enrollment)\n")
        rows = []
        for method in ("nexa", "nexa-noaug", "nexa-noaug-bnorm", "emb-only", "prototype", "mfcc-dtw"):
            for rule in ("deployed", "bank60-tail", "bank60+noisy", "oracle-wake"):
                a = mswc[(mswc.method == method) & (mswc.shots == 6) & (mswc.condition == "noisy")
                         & (mswc.target_fah == 1.0) & (mswc.rule == rule)]
                if not a.empty:
                    rows.append({"method": NAMES[method], "rule": RULES[rule], "FRR": a.frr.iloc[0],
                                 "FA/h mean": a.fah_mean.iloc[0], "FA/h median": a.fah_median.iloc[0]})
        frame = pd.DataFrame(rows)
        out.append("k = 6, target 1 FA/h (thresholds from clean enrollment, room audio and bank):\n")
        out.append(table(frame, [(c, c) for c in frame.columns]))
        out.append("")
        noise = R / "mswc" / "noise_only.csv"
        if noise.is_file():
            n = pd.read_csv(noise)
            g = n.groupby("method")[["noise_only_fah", "eval_noisy_fah", "eval_clean_fah"]].agg(["mean", "median"])
            g.columns = [f"{a} {b}" for a, b in g.columns]
            g = g.reset_index()
            out.append(f"Pure noise (no speech, {n.noise_hours.iloc[0] * 60:.0f} min, same level), seed 1, k = 6, "
                       "threshold from the clean bank at 1 FA/h (empirical):\n")
            out.append(table(g, [(c, c) for c in g.columns]))
            out.append("")
        cn = load("condnorm")
        if not cn.empty:
            out.append("Exploratory: condition-matched normaliser (seed 1, bank 60 min tail fit, target 1 FA/h):\n")
            rows = []
            for method in ("nexa-noaug-cn", "nexa-cn"):
                for k in (3, 6, 10):
                    for cond in ("clean", "noisy"):
                        a = cn[(cn.method == method) & (cn.shots == k) & (cn.condition == cond)
                               & (cn.target_fah == 1.0) & (cn.rule == "bank60-tail")]
                        if not a.empty:
                            rows.append({"method": NAMES[method], "k": k, "condition": cond, "FRR": a.frr.iloc[0],
                                         "FA/h mean": a.fah_mean.iloc[0], "FA/h median": a.fah_median.iloc[0],
                                         "wakes": int(a.wakes.iloc[0])})
            frame = pd.DataFrame(rows)
            out.append(table(frame, [(c, c) for c in frame.columns]))
            out.append("")

    fl = load("fleurs")
    if not fl.empty:
        info = common.load_json(R / "fleurs" / "streams" / "info.json")
        tasks = [common.load_json(p) for p in sorted((R / "fleurs" / "tasks").glob("*.json"))]
        masked = [t.get("fleurs_masked_utterances", 0) for t in tasks]
        hours = [t["negative_hours"]["fleurs"] for t in tasks]
        out.append("## 3b. Out-of-domain negatives: continuous read speech (FLEURS Persian test split)\n")
        out.append(f"- {info['utterances']} utterances, {info['seconds'] / 3600:.2f} h with 0.25 s gaps "
                   f"({info['speech_seconds'] / 3600:.2f} h of speech). Utterances containing the target word are "
                   f"masked for that wake: {min(masked)}-{max(masked)} per wake, leaving "
                   f"{min(hours):.2f}-{max(hours):.2f} h.")
        out.append("- Same wakes (cross-speaker, k = 6, 48 wakes) and the same thresholds as section 1; FRR is the "
                   "MSWC positive FRR at that threshold.\n")
        rows = []
        for method in ("nexa", "nexa-noaug", "nexa-noaug-bnorm", "emb-only", "mfcc-dtw"):
            for rule in ("deployed", "global-bank", "bank60", "bank60-os", "bank60-tail", "oracle-wake"):
                a = fl[(fl.method == method) & (fl.condition == "clean") & (fl.target_fah == 1.0) & (fl.rule == rule)]
                if a.empty:
                    continue
                r = a.iloc[0]
                rows.append({"method": NAMES[method], "rule": RULES[rule], "FRR (MSWC positives)": r.frr,
                             "FA/h FLEURS mean": r.fah_mean, "FA/h FLEURS median": r.fah_median,
                             "share of wakes <= 2 FA/h": r.share_within_2x,
                             "FA/h MSWC eval (same threshold)": r.get("fah_eval_clean_mean", np.nan)})
        frame = pd.DataFrame(rows)
        out.append("Target 1 FA/h (oracle per wake = threshold chosen on FLEURS itself):\n")
        out.append(table(frame, [(c, c) for c in frame.columns]))
        out.append("")

    same = load("samespk")
    if not same.empty:
        plan = common.load_json(R / "samespk" / "plan.json")
        main_tasks = [t for t in plan if t["set"] == "main" and t["shots"] == 10]
        broad_tasks = [t for t in plan if t["set"] == "broad"]
        out.append("## 4. Same-speaker enrollment (MSWC)\n")
        out.append(f"- Main set: {len(main_tasks)} evaluation speakers with a word repeated >= 15 times "
                   f"(one word per speaker; words: "
                   + ", ".join(f"{w} x{c}" for w, c in pd.Series([t['word'] for t in main_tasks]).value_counts().items())
                   + f"); k = 3, 6, 10 nested; {sum(len(t['positives']) for t in main_tasks)} positives in total "
                   f"(5-20 per speaker).")
        out.append(f"- Broad set (k = 3): {len(broad_tasks)} speakers with >= 8 repeats, "
                   f"{sum(len(t['positives']) for t in broad_tasks)} positives (enroll 3, test on up to 20 others); "
                   f"words: " + ", ".join(f"{w} x{c}" for w, c in
                                           pd.Series([t['word'] for t in broad_tasks]).value_counts().items()) + ".")
        full = pd.read_csv(common.MSWC / "test.csv", usecols=["SPEAKER", "WORD", "GENDER"])
        pairs = full.groupby(["SPEAKER", "WORD"]).size()
        avail = []
        for m in (8, 15, 20, 30):
            q = pairs[pairs >= m]
            avail.append(f">= {m}: {len(q)} pairs ({q.index.get_level_values(0).nunique()} speakers, "
                         f"{q.index.get_level_values(1).nunique()} words)")
        out.append("- What MSWC offers: speaker-word pairs with " + "; ".join(avail) + ". We use one word per "
                   "speaker (the most repeated), so the speakers are independent. The repeated words are mostly very "
                   "frequent one- or two-syllable function words (این, است and آیا are "
                   f"{sum(t['word'] in ('این', 'است', 'آیا') for t in main_tasks)} of the {len(main_tasks)} main-set "
                   "words) and only partly overlap the cross-speaker targets, so the two settings differ in the words "
                   "as well as in the speakers. The repeats of a speaker are "
                   "cut from different Common Voice sentences read by that speaker; enrollment and test may share "
                   "session and microphone, unlike a real far-field room.")
        genders = full.drop_duplicates("SPEAKER").GENDER.value_counts()
        out.append(f"- Every speaker in both sets is labelled "
                   f"{', '.join(sorted(set(t['gender'] for t in broad_tasks)))} in MSWC (all {full.SPEAKER.nunique():,} "
                   f"test speakers: " + ", ".join(f"{g} {n:,}" for g, n in genders.items()) + ").")
        table_words = pd.read_csv(common.MSWC / "test.csv", usecols=["LINK", "WORD"])
        link_word = dict(zip(table_words.LINK, table_words.WORD))
        own_total = sum(len(t["own"]) for t in main_tasks)
        own_sub = sum(t["word"] in link_word[link] for t in main_tasks for link in t["own"])
        out.append(f"- Own-voice negatives: the same speaker saying other words (outside the bank vocabulary, up to "
                   f"400 clips per speaker); FA/h pooled over speakers. In the main set {own_sub} of {own_total} "
                   f"own-voice clips are words that contain the target as a substring (e.g. اینجا for این); they "
                   f"are kept as negatives.")
        out.append("- Other-speaker negatives: the 4 h seed-1 evaluation stream of section 1.")
        enroll_csv = R / "tables" / "samespk_enrollment.csv"
        if enroll_csv.is_file():
            e = pd.read_csv(enroll_csv)
            out.append("- A wake needs at least 3 accepted enrollment recordings (as in NexaRoom). Wakes built / "
                       "planned: " + "; ".join(
                           f"{r.set} k = {r.shots}: {r.built}/{r.planned}"
                           + (f" ({r.finished} finished)" if r.finished < r.planned else "")
                           for r in e.sort_values(["set", "shots"], ascending=[False, True]).itertuples())
                       + ". Wakes that could not be built are left out of the averages below.\n")
        rows = []
        for (setname, k) in (("main", 10), ("broad", 3)):
            for method in ("nexa", "nexa-min", "nexa-noaug", "nexa-noaug-bnorm", "emb-only", "prototype", "mfcc-dtw"):
                for rule in ("deployed", "global-bank", "bank60-tail", "oracle-wake"):
                    a = same[(same.set == setname) & (same.shots == k) & (same.method == method)
                             & (same.condition == "clean") & (same.target_fah == 1.0) & (same.rule == rule)]
                    if a.empty:
                        continue
                    r = a.iloc[0]
                    rows.append({"set": setname, "k": k, "method": NAMES[method], "rule": RULES[rule],
                                 "speakers": int(r.wakes), "FRR [95% CI]": fmt_ci(r), "FA/h other speakers": r.fah_mean,
                                 "FA/h own voice (pooled)": r.get("fah_own_clean_pooled", np.nan),
                                 "own-voice hours": r.get("hours_own_clean_total", np.nan)})
        frame = pd.DataFrame(rows)
        out.append("Clean condition, target 1 FA/h (main set at k = 10 and broad set at k = 3; main set at k = 3 and 6 "
                   "in the k table below and in `results/tables/samespk_rules.csv`). 'oracle per wake' is set on the "
                   "other-speaker negatives; own-voice FA/h is measured at the same threshold:\n")
        out.append(table(frame, [(c, c) for c in frame.columns]))
        out.append("")
        spm_path = R / "samespk" / "paired_methods.csv"
        if spm_path.is_file() and spm_path.stat().st_size > 1:
            spm = pd.read_csv(spm_path)
            spm = spm[(spm.condition == "clean") & (spm.target_fah == 1.0)
                      & spm.rule.isin(["oracle-wake", "bank60-tail"])]
            rows = []
            for (a, b), frame in spm.groupby(["a", "b"], sort=False):
                row = {"method A": NAMES[a], "method B": NAMES[b]}
                for setname, k in (("main", 10), ("broad", 3)):
                    for rule in ("oracle-wake", "bank60-tail"):
                        r = frame[(frame.set == setname) & (frame.shots == k) & (frame.rule == rule)]
                        if r.empty:
                            continue
                        r = r.iloc[0]
                        short = {"oracle-wake": "oracle/wake", "bank60-tail": "bank tail"}[rule]
                        row[f"{setname} k={k}, {short}"] = (f"{r.d_frr:+.3f} [{r.d_frr_lo:+.3f}, {r.d_frr_hi:+.3f}] "
                                                            f"(n={int(r.wakes)})")
                rows.append(row)
            frame = pd.DataFrame(rows)
            out.append("Paired method differences, same speaker, FRR(A) - FRR(B) on the same wakes (clean, target "
                       "1 FA/h; 95 % bootstrap over speakers):\n")
            out.append(table(frame, [(c, c) for c in frame.columns]))
            out.append("")
        rows = []
        for k in (3, 6, 10):
            for method in ("nexa", "nexa-noaug", "mfcc-dtw"):
                for rule in ("bank60-tail", "oracle-wake"):
                    a = same[(same.set == "main") & (same.shots == k) & (same.method == method)
                             & (same.condition == "noisy") & (same.target_fah == 1.0) & (same.rule == rule)]
                    if not a.empty:
                        rows.append({"k": k, "method": NAMES[method], "rule": RULES[rule], "FRR": a.frr.iloc[0],
                                     "FA/h": a.fah_mean.iloc[0]})
        frame = pd.DataFrame(rows)
        out.append("Noisy condition (main set), target 1 FA/h:\n")
        out.append(table(frame, [(c, c) for c in frame.columns]))
        out.append("")
        paired_k = R / "tables" / "samespk_paired_k.csv"
        if paired_k.is_file() and paired_k.stat().st_size > 1:
            pk = pd.read_csv(paired_k)
            pk = pk[(pk.target_fah == 1.0) & pk.rule.isin(["deployed", "bank60-tail", "oracle-wake"])
                    & pk.method.isin(["nexa", "nexa-noaug", "emb-only", "prototype", "mfcc-dtw"])]
            ci_path = R / "samespk" / "paired_k.csv"
            ci = pd.read_csv(ci_path) if ci_path.is_file() and ci_path.stat().st_size > 1 else pd.DataFrame()
            rows = []
            for (method, rule), frame in pk.groupby(["method", "rule"], sort=False):
                row = {"method": NAMES[method], "rule": RULES[rule], "speakers": int(frame.speakers.iloc[0])}
                for r in frame.sort_values("shots").itertuples():
                    row[f"k={r.shots} FRR / FA/h other / FA/h own"] = (f"{r.frr:.3f} / {r.fah_other_speakers:.2f} / "
                                                                       f"{r.fah_own_pooled:.2f}")
                if not ci.empty:
                    c = ci[(ci.method == method) & (ci.rule == rule) & (ci.condition == "clean")
                           & (ci.target_fah == 1.0) & (ci.a == 10) & (ci.b == 3)]
                    if not c.empty:
                        c = c.iloc[0]
                        row["dFRR k10-k3 [95% CI]"] = f"{c.d_frr:+.3f} [{c.d_frr_lo:+.3f}, {c.d_frr_hi:+.3f}]"
                rows.append(row)
            frame = pd.DataFrame(rows)
            out.append("Main set, only speakers whose wake was built at all three k (paired over k), clean, "
                       "target 1 FA/h:\n")
            out.append(table(frame, [(c, c) for c in frame.columns]))
            out.append("")

    esc = load("esc50")
    if not esc.empty:
        info = common.load_json(R / "esc50" / "streams" / "info.json")
        out.append("## 5. Non-speech wakes (ESC-50, all 50 classes)\n")
        out.append(f"- Enrollment: k of the 8 fold-1 clips of the class (loudest 1.5 s); positives: the 32 clips of the "
                   f"class in folds 2-5; negatives: folds 3-5 of the other classes ({info['eval']['seconds'] / 3600:.2f} h "
                   f"before masking the wake's own clips) and the 4 h seed-1 speech stream; sound bank: folds 1-2 "
                   f"({info['bank']['seconds'] / 3600:.2f} h before masking). 3 seeds x 50 classes per k.\n")
        rows = []
        for k in (3, 6):
            for method in ("nexa", "nexa-noaug", "nexa-noaug-bnorm", "emb-only", "mel-only", "prototype", "mfcc-dtw"):
                for rule in ("deployed", "speechbank-tail", "speech+soundbank", "oracle-wake"):
                    a = esc[(esc.shots == k) & (esc.method == method) & (esc.target_fah == 1.0) & (esc.rule == rule)]
                    if a.empty:
                        continue
                    r = a.iloc[0]
                    rows.append({"k": k, "method": NAMES[method], "rule": RULES[rule], "FRR [95% CI]": fmt_ci(r),
                                 "FA/h sounds": r.get("fah_sound_eval_mean", np.nan),
                                 "FA/h speech": r.get("fah_speech_eval_mean", np.nan)})
        frame = pd.DataFrame(rows)
        out.append("Target 1 FA/h (FA/h measured on sounds + speech together):\n")
        out.append(table(frame, [(c, c) for c in frame.columns]))
        out.append("")
        per = pd.read_csv(R / "esc50" / "per_task.csv")
        sel = per[(per.method == "nexa-noaug") & (per.shots == 6) & (per.target_fah == 1.0)
                  & (per.rule == "speech+soundbank")]
        cls = sel.groupby("category").agg(frr=("frr", "mean"), fah=("fah", "mean")).sort_values("frr").reset_index()
        cls.to_csv(R / "esc50" / "per_class_noaug_k6.csv", index=False)
        out.append("Per class (clean templates, k = 6, speech + sound bank, 1 FA/h), best and worst 10 "
                   "(full list in `results/esc50/per_class_noaug_k6.csv`):\n")
        out.append(table(pd.concat([cls.head(10), cls.tail(10)]), [("category", "class"), ("frr", "FRR"),
                                                                   ("fah", "FA/h")]))
        out.append("")
        chosen = ["clapping", "door_wood_knock", "mouse_click", "keyboard_typing", "can_opening", "clock_alarm",
                  "glass_breaking", "coughing", "sneezing", "laughing"]
        sel = per[per.category.isin(chosen) & (per.target_fah == 1.0) & per.method.isin(["nexa", "nexa-noaug"])
                  & per.rule.isin(["deployed", "speechbank-tail", "speech+soundbank"])]
        g = sel.groupby(["category", "method", "rule", "shots"]).agg(frr=("frr", "mean"), fah=("fah", "mean"),
                                                                     seeds=("seed", "nunique")).reset_index()
        piv = g.pivot_table(index=["category"], columns=["method", "rule", "shots"], values=["frr", "fah"])
        rows = []
        for cat in chosen:
            if cat not in piv.index:
                continue
            row = {"class": cat}
            for method in ("nexa", "nexa-noaug"):
                for rule in ("deployed", "speech+soundbank"):
                    for k in (3, 6):
                        try:
                            f, a = piv.loc[cat, ("frr", method, rule, k)], piv.loc[cat, ("fah", method, rule, k)]
                        except KeyError:
                            continue
                        short = {"nexa": "deployed tmpl", "nexa-noaug": "clean tmpl"}[method]
                        rname = {"deployed": "room cal.", "speech+soundbank": "bank cal."}[rule]
                        row[f"{short}, {rname}, k={k}: FRR / FA/h"] = f"{f:.2f} / {a:.1f}"
            rows.append(row)
        frame = pd.DataFrame(rows)
        out.append("Sounds a user could make on purpose (mean over 3 seeds; FA/h on sounds + speech; "
                   "bank cal. = speech + sound bank, empirical, 1 FA/h):\n")
        out.append(table(frame, [(c, c) for c in frame.columns]))
        out.append("")

    xc_path = R / "esc50_xclass" / "calibration.csv"
    if xc_path.is_file():
        xc = pd.read_csv(xc_path)
        xp = pd.read_csv(R / "esc50_xclass" / "paired.csv")
        xinfo = common.load_json(R / "esc50_xclass" / "streams" / "info.json")
        hours = {k: v["seconds"] / 3600 for k, v in xinfo.items() if isinstance(v, dict) and "seconds" in v}
        out.append("## 5b. Non-speech wakes: does the sound bank need the same sound types? (ESC-50, class-disjoint)\n")
        out.append(f"- The 50 classes are split once into two halves of 25 (five per ESC-50 major group, seed "
                   f"{xinfo['split_seed']}). Bank streams (folds 1-2): {hours['H1_bank']:.2f} h and "
                   f"{hours['H2_bank']:.2f} h; evaluation streams (folds 3-5): {hours['H1_eval']:.2f} h and "
                   f"{hours['H2_eval']:.2f} h, before masking the wake's own clips.")
        out.append("- *seen*: bank and evaluation sounds from the same half (different recordings); *unseen*: bank from "
                   "one half, evaluation sounds from the other. Both use the 60-min speech bank and the 4 h speech "
                   "stream as well, and the same wakes, templates and positives as section 5. Empirical rule; each "
                   "condition is the mean of its two half pairings.\n")
        rows = []
        for k in sorted(xc.shots.unique(), reverse=True):
            for method in ("nexa-noaug", "nexa"):
                for cond in ("seen", "unseen"):
                    a = xc[(xc.shots == k) & (xc.method == method) & (xc.condition == cond) & (xc.target_fah == 1.0)]
                    if a.empty:
                        continue
                    r = a.iloc[0]
                    rows.append({"k": k, "method": NAMES[method], "bank sounds": cond, "wakes": int(r.wakes),
                                 "FRR [95% CI]": fmt_ci(r), "FA/h mean": r.fah_mean,
                                 "FA/h median": r.fah_median, "FA/h sounds only": r.fah_sound_mean,
                                 "wakes <= 2 FA/h": r.share_within_2x})
        out.append("Target 1 FA/h (FA/h on speech + sound negatives together unless stated):\n")
        out.append(table(pd.DataFrame(rows), [(c, c) for c in rows[0]] if rows else []))
        out.append("")
        sel = xp[xp.target_fah == 1.0]
        rows = [{"k": r.shots, "method": NAMES[r.method], "wakes": int(r.wakes),
                 "dFA/h [95% CI]": f"{r.d_fah:+.2f} [{r.d_fah_lo:+.2f}, {r.d_fah_hi:+.2f}]",
                 "dFA/h sounds only [95% CI]": f"{r.d_fah_sound:+.2f} [{r.d_fah_sound_lo:+.2f}, "
                                               f"{r.d_fah_sound_hi:+.2f}]"}
                for r in sel.sort_values(["shots", "method"], ascending=[False, False]).itertuples()]
        if rows:
            out.append("Paired difference unseen minus seen (same wakes, cluster bootstrap over classes), "
                       "target 1 FA/h. FRR is identical in the two conditions by construction: both use the "
                       "thresholds of the same two banks, only the evaluation half is swapped.\n")
            out.append(table(pd.DataFrame(rows), [(c, c) for c in rows[0]]))
            out.append("")

    ss = load("esc50_samesrc")
    if not ss.empty:
        plan = common.load_json(R / "esc50_samesrc" / "plan.json")
        sper = pd.read_csv(R / "esc50_samesrc" / "per_task.csv")
        recs = {(t["category"], t["src_file"]) for t in plan}
        npos = sum(len(t["positives"]) for t in plan)
        out.append("## 5c. Non-speech wakes, same-source enrollment (ESC-50 takes of one recording)\n")
        out.append(f"- {len(recs)} recordings with >= 4 takes, from {len({c for c, _ in recs})} classes; k = 3 takes "
                   f"enrolled, the other takes of the same recording are the positives; up to 3 distinct enrollment "
                   f"subsets per recording: {len(plan)} wakes, {npos} positive trials in total (1-5 per wake). "
                   "Negatives and banks as in section 5; 95% intervals: cluster bootstrap over recordings.")
        out.append("- The cross-recording rows use the main k = 3 run restricted to the same classes "
                   "(other recordings of the class as positives).\n")
        esc_per = R / "esc50" / "per_task.csv"
        cross = pd.read_csv(esc_per) if esc_per.is_file() else pd.DataFrame()
        cats = sorted({c for c, _ in recs})
        rows = []
        for method in ("nexa", "nexa-noaug", "nexa-noaug-bnorm", "emb-only", "prototype", "mfcc-dtw"):
            for rule in ("deployed", "speechbank-tail", "speech+soundbank", "speech+soundbank-tail", "oracle-wake"):
                a = ss[(ss.method == method) & (ss.target_fah == 1.0) & (ss.rule == rule)]
                if a.empty:
                    continue
                r = a.iloc[0]
                row = {"method": NAMES[method], "rule": RULES.get(rule, rule), "wakes": int(r.wakes),
                       "same source: FRR [95% CI]": fmt_ci(r), "same source: FA/h mean": r.fah_mean,
                       "same source: FA/h median": r.fah_median}
                if not cross.empty:
                    c = cross[(cross.shots == 3) & (cross.method == method) & (cross.target_fah == 1.0)
                              & (cross.rule == rule) & cross.category.isin(cats)]
                    if not c.empty:
                        row["cross-recording, same classes: FRR"] = c.frr.mean()
                        row["cross-recording: FA/h mean"] = c.fah.mean()
                rows.append(row)
        if rows:
            out.append("k = 3, target 1 FA/h (FA/h on sounds + speech):\n")
            out.append(table(pd.DataFrame(rows), [(c, c) for c in rows[0]]))
            out.append("")
        sel = sper[(sper.method == "nexa-noaug") & (sper.target_fah == 1.0) & (sper.rule == "speech+soundbank")]
        if not sel.empty:
            cls = sel.groupby("category").agg(wakes=("frr", "size"), frr=("frr", "mean"),
                                               fah=("fah", "mean")).reset_index().sort_values("frr")
            cls.to_csv(R / "esc50_samesrc" / "per_class_noaug.csv", index=False)
            out.append("Per class, clean templates, speech + sound bank (empirical), 1 FA/h "
                       "(`results/esc50_samesrc/per_class_noaug.csv`):\n")
            out.append(table(cls, [("category", "class"), ("wakes", "wakes"), ("frr", "FRR"), ("fah", "FA/h")]))
            out.append("")

    tables = _cost_tables()
    if tables is not None:
        info, c, md, b = tables
        f = info["features"]
        out.append("## 6. Compute cost (single thread, x86 virtual machine; Raspberry Pi not measured)\n")
        out.append(f"- Machine: {info['cpu']} host, {info.get('hypervisor') or 'no'} virtual machine with "
                   f"{info.get('logical_cpus', '?')} vCPUs; the script was pinned to vCPU {info['affinity']} and "
                   f"used one thread (ONNX Runtime intra/inter-op threads = 1, BLAS and Numba threads = 1). Other "
                   f"jobs were running (load average {info['load_average_before'][0]:.1f} before, "
                   f"{info['load_average_after'][0]:.1f} after, on 20 vCPUs), so the times are upper bounds for "
                   f"this CPU. Clock: thread CPU time; every measurement repeated {info['repeats']} times, the "
                   f"tables give the median over repeats and, in brackets, the smallest and largest repeat.")
        out.append(f"- Software: Python {info['python']}, NumPy {info['numpy']}, ONNX Runtime "
                   f"{info['onnxruntime']}, Numba {info['numba']}.")
        out.append(f"- Feature extraction per 80 ms frame (deployed `FeatureStream.push`, {f['frames_per_repeat']} "
                   f"frames per repeat): median {f['feature_ms_median']:.3f} ms "
                   f"[{f['feature_ms_median_min']:.3f}, {f['feature_ms_median_max']:.3f}], p95 "
                   f"{f['feature_ms_p95']:.3f} ms, wall-clock median {f['feature_ms_median_wall']:.3f} ms; of this "
                   f"the log-mel model takes {f['mel_ms_median']:.3f} ms and the embedding model "
                   f"{f['embedding_ms_median']:.3f} ms.")
        out.append(f"- Template length in real enrollments (seed 1, k = 10, 16 words): median "
                   f"{info['template_length_median']} frames of 80 ms, p90 {info['template_length_p90']}, max "
                   f"{info['template_length_max']} ({info['template_lengths_n']} templates; the deployed cap is 30). "
                   f"MFCC + DTW templates: median {info['mfcc_template_length_median']} frames of 10 ms "
                   f"(p90 {info['mfcc_template_length_p90']}). Raw lengths: `results/cost/template_lengths.csv`.")
        out.append("- Templates per wake in practice: the deployed enrollment keeps 3 templates per recording "
                   "(clean + two noise copies), i.e. 9, 18 and 30 for k = 3, 6 and 10; clean templates only: k. "
                   "Several wakes on one device add their templates.\n")
        out.append("Per-frame cost of matching (random templates of the given length; the cost does not depend on "
                   "the values). 'Deployed' = `WakeDetector.process` (NumPy `TemplateMatcher.step`, `wake_scores` "
                   "and the decision layer) as it runs on the device; 'compiled' = the same recurrence in Numba "
                   "(`nexawake/fast.py`), amortised over 1,000 frames:\n")

        def rng(row, col, fmt="{:.3f}"):
            return f"{fmt.format(row[col])} [{fmt.format(row[col + '_min'])}, {fmt.format(row[col + '_max'])}]"

        c2 = c.sort_values(["length", "templates"]).copy()
        c2["dep"] = c2.apply(lambda r: rng(r, "detector_ms_median"), axis=1)
        c2["comp"] = c2.apply(lambda r: rng(r, "compiled_ms", "{:.4f}"), axis=1)
        c2["rtf"] = c2.real_time_factor.map(lambda v: f"{100 * v:.2f} %")
        short, long_ = sorted(c.length.unique())[0], sorted(c.length.unique())[-1]
        cs = c.set_index(["length", "templates"])
        out.append(f"The deployed NumPy matcher loops in Python over the template columns, so its time depends "
                   f"mainly on the longest template: with {short}-frame templates it grows from "
                   f"{cs.loc[(short, 3), 'detector_ms_median']:.2f} ms (3 templates) to "
                   f"{cs.loc[(short, 240), 'detector_ms_median']:.2f} ms (240 templates), while at 18 templates "
                   f"{long_}-frame templates take {cs.loc[(long_, 18), 'detector_ms_median']:.2f} ms. The compiled "
                   f"matcher grows roughly in proportion to templates x length "
                   f"({cs.loc[(short, 18), 'compiled_ms']:.4f} ms for 18 x {short}, "
                   f"{cs.loc[(long_, 240), 'compiled_ms']:.3f} ms for 240 x {long_}).\n")
        out.append(table(c2, [("templates", "templates"), ("length", "frames per template"),
                              ("dep", "deployed ms/frame, median [range]"), ("detector_ms_p95", "deployed p95 ms"),
                              ("comp", "compiled ms/frame [range]"),
                              ("total_ms_with_features", "features + deployed, ms/frame"),
                              ("rtf", "share of real time (features + deployed)"),
                              ("total_ms_compiled", "features + compiled, ms/frame")]))
        out.append("")
        out.append(f"MFCC + DTW baseline per 80 ms of audio (8 MFCC frames; MFCC with librosa amortised over "
                   f"{info['mfcc_features']['block_seconds']:.0f} s blocks: "
                   f"{info['mfcc_features']['mfcc_ms_per_80ms']:.3f} ms "
                   f"[{info['mfcc_features']['mfcc_ms_per_80ms_min']:.3f}, "
                   f"{info['mfcc_features']['mfcc_ms_per_80ms_max']:.3f}]; DTW compiled), next to the embedding "
                   f"matcher with the same number of templates:\n")
        typical = info["template_length_median"]
        emb = c[c.length == typical].set_index("templates")
        md2 = md.sort_values("templates").copy()
        md2["dtw"] = md2.apply(lambda r: rng(r, "compiled_ms_per_80ms"), axis=1)
        md2["emb"] = md2.templates.map(lambda n: emb.loc[n, "total_ms_compiled"] if n in emb.index else np.nan)
        out.append(table(md2, [("templates", "templates"), ("length", "frames per template (10 ms)"),
                               ("dtw", "compiled DTW ms per 80 ms [range]"),
                               ("total_ms_per_80ms", "MFCC + DTW total, ms per 80 ms"),
                               ("emb", f"features + compiled matcher ({typical}-frame templates), ms")]))
        out.append("")
        out.append(f"Bank calibration for one wake: scoring the 60-min bank ({int(b.bank_frames.iloc[0]):,} frames) "
                   f"and counting false-alarm events at all {int(b.grid_thresholds.iloc[0])} grid thresholds "
                   f"(templates of {typical} frames; deployed NumPy matcher measured once):\n")
        b2 = b.sort_values("templates").copy()
        b2["comp"] = b2.apply(lambda r: rng(r, "compiled_seconds", "{:.2f}"), axis=1)
        b2["ev"] = b2.apply(lambda r: rng(r, "event_count_seconds", "{:.2f}"), axis=1)
        cols = [("templates", "templates"), ("comp", "compiled matcher, s [range]"), ("ev", "event counting, s")]
        if "deployed_seconds" in b2:
            b2["dep"] = b2.deployed_seconds.map(lambda v: "-" if not np.isfinite(v) else f"{v:.1f}")
            cols.append(("dep", "deployed NumPy matcher, s"))
        out.append(table(b2, cols))
        out.append("")
        out.append("Raw per-frame times: `results/cost/raw_times.npz`; per-repeat rows: `results/cost/raw_*.csv`; "
                   "machine and software: `results/cost/cost_info.json`.\n")
    out.append("## 7. Tables and figures for the manuscript\n")
    out.append("Every table above is generated by `experiments/make_report.py` from the CSV files below; nothing is "
               "typed in by hand. Run `experiments/run_analysis.sh` to rebuild all of them from `results/*/tasks/`.\n")
    files = [
        ("tables/negatives.csv", "length of every negative stream and the FA/h resolution"),
        ("tables/mswc_rules.csv", "cross-speaker: method x k x condition x target x threshold rule (sections 1-3)"),
        ("tables/mswc_shots.csv", "cross-speaker: FRR and FA/h by method and k at the bank tail-fit threshold (section 2)"),
        ("mswc/paired.csv, mswc/paired_methods.csv, mswc/paired_k.csv", "paired differences with bootstrap intervals"),
        ("mswc/noise_only.csv", "pure-noise diagnostic (section 3)"),
        ("tables/condnorm_rules.csv", "condition-matched normaliser, exploratory (section 3)"),
        ("tables/fleurs_rules.csv", "FLEURS read-speech negatives (section 3b)"),
        ("tables/samespk_rules.csv, tables/samespk_enrollment.csv, tables/samespk_paired_k.csv",
         "same-speaker enrollment (section 4)"),
        ("tables/esc50_rules.csv, esc50/per_class_noaug_k6.csv", "ESC-50 sound wakes (section 5)"),
        ("tables/esc50_xclass.csv", "class-disjoint sound bank (section 5b)"),
        ("tables/esc50_samesrc_rules.csv, esc50_samesrc/per_class_noaug.csv", "same-source sound wakes (section 5c)"),
        ("tables/cost_matcher.csv, tables/cost_mfcc_dtw.csv, tables/cost_bank_calibration.csv, cost/cost_info.json",
         "compute cost (section 6)"),
    ]
    out.append(table(pd.DataFrame(files, columns=["file", "content"]),
                     [("file", "file (under results/)"), ("content", "content")]))
    out.append("")
    figs = [
        ("fig_det_k6", "mswc/det.csv, mswc/calibration.csv",
         "pooled DET curves, cross-speaker k = 6, clean and 10 dB noise; markers: deployed and bank calibration"),
        ("fig_calibration", "mswc/per_task.csv",
         "achieved FA/h per wake for every threshold rule, clean templates, target 1 FA/h, k = 3, 6, 10 pooled"),
        ("fig_shots", "mswc/calibration.csv, samespk/calibration.csv",
         "FRR vs k at the bank tail-fit threshold, cross-speaker and same speaker (own-voice FA/h printed)"),
        ("fig_esc50_classes", "esc50/per_class_noaug_k6.csv", "detection rate per ESC-50 class, k = 6"),
        ("fig_cost", "cost/matcher_cost.csv, cost/cost_info.json", "per-frame matching cost vs templates"),
    ]
    if not (R / "cost" / "matcher_cost.csv").is_file():
        figs = [f for f in figs if f[0] != "fig_cost"]
    out.append("Figures (`results/figures/`, PNG at 200 dpi and PDF, made by `experiments/figures.py` after this "
               "report, which writes the per-class CSVs they read):\n")
    out.append(table(pd.DataFrame(figs, columns=["figure", "source", "content"]),
                     [("figure", "figure"), ("source", "data (under results/)"), ("content", "content")]))
    out.append("")
    (R / "summary.md").write_text("\n".join(out) + "\n")
    print((R / "summary.md").read_text()[:3000])


if __name__ == "__main__":
    main()
