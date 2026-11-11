"""Compact summary tables (CSV) for the manuscript, read from results/*/calibration.csv,
per_task.csv and the raw task files. Nothing here recomputes scores.

Writes results/tables/:
  mswc_rules.csv            cross-speaker: every method x k x condition x target for the main threshold rules
  mswc_shots.csv            cross-speaker: FRR / FA/h by method and k at the bank 60 min tail-fit threshold
  samespk_rules.csv         same speaker: the same, plus own-voice FA/h (pooled over speakers)
  samespk_enrollment.csv    same speaker: how many wakes could be built per set and k
  samespk_paired_k.csv      same speaker, main set: speakers whose wake was built at k = 3, 6 and 10
  negatives.csv             lengths of every negative stream used (hours) and the FA/h resolution
  esc50_rules.csv           ESC-50 sound wakes (cross-recording): method x k x rule x target, FA/h split into
                            sounds and speech
  esc50_samesrc_rules.csv   ESC-50 same-source wakes (k = 3), the same columns
  esc50_xclass.csv          ESC-50 class-disjoint sound bank: seen vs unseen bank sound types
  fleurs_rules.csv          cross-speaker wakes (k = 6) on FLEURS read speech, MSWC thresholds
  condnorm_rules.csv        exploratory condition-matched normaliser (seed 1)
  cost_*.csv                copies of results/cost/{matcher,mfcc_dtw,bank_calibration}_cost.csv (experiments/cost.py)

Usage: python experiments/export_tables.py
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
OUT = R / "tables"
RULES = ["deployed", "global-bank", "bank15", "bank30", "bank60", "bank60-os", "bank15-tail", "bank30-tail",
         "bank60-tail", "bank60+noisy", "oracle-global", "oracle-wake"]
COLS = ["wakes", "clusters", "frr", "frr_lo", "frr_hi", "fah_mean", "fah_lo", "fah_hi", "fah_median", "fah_p90",
        "fah_max", "share_within_target", "share_within_2x", "tau_median"]


def tasks(name: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((R / name / "tasks").glob("*.json"))]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cal = pd.read_csv(R / "mswc" / "calibration.csv")
    sel = cal[cal.rule.isin(RULES)]
    sel[["shots", "method", "condition", "rule", "target_fah"] + COLS].sort_values(
        ["condition", "method", "shots", "target_fah", "rule"]).to_csv(OUT / "mswc_rules.csv", index=False)

    rows = []
    for (method, cond, k), frame in cal[(cal.target_fah == 1.0)].groupby(["method", "condition", "shots"]):
        get = lambda rule, col: frame[frame.rule == rule][col].iloc[0] if (frame.rule == rule).any() else np.nan  # noqa: E731
        rows.append({"method": method, "condition": cond, "shots": k, "wakes": get("bank60-tail", "wakes"),
                     "frr_bank60_tail": get("bank60-tail", "frr"), "frr_lo": get("bank60-tail", "frr_lo"),
                     "frr_hi": get("bank60-tail", "frr_hi"), "fah_bank60_tail": get("bank60-tail", "fah_mean"),
                     "frr_deployed": get("deployed", "frr"), "fah_deployed": get("deployed", "fah_mean"),
                     "frr_global_bank": get("global-bank", "frr"), "fah_global_bank": get("global-bank", "fah_mean"),
                     "frr_oracle_global": get("oracle-global", "frr"),
                     "frr_oracle_wake": get("oracle-wake", "frr")})
    pd.DataFrame(rows).to_csv(OUT / "mswc_shots.csv", index=False)

    neg = []
    for seed in (1, 2, 3):
        info = common.load_json(R / "mswc" / "streams" / f"seed{seed}" / "info.json")
        neg.append({"experiment": "mswc", "stream": "eval_clean", "seed": seed, "clips": info["negative_clips"],
                    "hours": info["clean_seconds"] / 3600, "frames": info["clean_frames"]})
        neg.append({"experiment": "mswc", "stream": "eval_noisy", "seed": seed, "clips": info["negative_clips"],
                    "hours": info["noisy_seconds"] / 3600, "frames": info["noisy_frames"]})
        neg.append({"experiment": "mswc", "stream": "room", "seed": seed, "clips": 20,
                    "hours": info["room_seconds"] / 3600, "frames": None})
    bank = common.load_json(R / "bank" / "bank_info.json")
    neg.append({"experiment": "all", "stream": "bank_clean", "seed": None, "clips": bank["clips"],
                "hours": bank["seconds"] / 3600, "frames": bank["clean_frames"], "MB_fp16": bank["clean_feat_MB_fp16"]})
    neg.append({"experiment": "all", "stream": "bank_noisy", "seed": None, "clips": bank["clips"],
                "hours": bank["seconds"] / 3600, "frames": bank["noisy_frames"], "MB_fp16": bank["noisy_feat_MB_fp16"]})
    esc = R / "esc50" / "streams" / "info.json"
    if esc.is_file():
        info = common.load_json(esc)
        for name, stream in (("eval", "sound_eval (folds 3-5, all classes)"), ("bank", "sound_bank (folds 1-2, all classes)")):
            neg.append({"experiment": "esc50", "stream": stream, "seed": None, "clips": info[name]["clips"],
                        "hours": info[name]["seconds"] / 3600, "frames": info[name]["frames"],
                        "MB_fp16": info[name]["frames"] * 128 * 2 / 1e6 if name == "bank" else None})
    xc = R / "esc50_xclass" / "streams" / "info.json"
    if xc.is_file():
        info = common.load_json(xc)
        for half in ("H1", "H2"):
            for name in ("eval", "bank"):
                v = info[f"{half}_{name}"]
                neg.append({"experiment": "esc50_xclass", "stream": f"{half}_{name}", "seed": None, "clips": v["clips"],
                            "hours": v["seconds"] / 3600, "frames": v["frames"]})
    fl = R / "fleurs" / "streams" / "info.json"
    if fl.is_file():
        info = common.load_json(fl)
        neg.append({"experiment": "fleurs", "stream": "fleurs fa_ir test (continuous read speech)", "seed": None,
                    "clips": info["utterances"], "hours": info["seconds"] / 3600, "frames": info["frames"]})
    noise = R / "mswc" / "noise_only.csv"
    if noise.is_file():
        n = pd.read_csv(noise)
        neg.append({"experiment": "mswc noise diagnostic", "stream": "pure white + pink noise", "seed": 1,
                    "clips": None, "hours": float(n.noise_hours.iloc[0]), "frames": None})
    neg = pd.DataFrame(neg)
    neg["fah_per_event"] = 1.0 / neg.hours
    neg.to_csv(OUT / "negatives.csv", index=False)

    scal = R / "samespk" / "calibration.csv"
    if scal.is_file():
        same = pd.read_csv(scal)
        extra = [c for c in same.columns if c.startswith(("fah_own", "hours_own"))]
        same[same.rule.isin(RULES)][["set", "shots", "method", "condition", "rule", "target_fah"] + COLS + extra] \
            .sort_values(["set", "condition", "method", "shots", "target_fah", "rule"]) \
            .to_csv(OUT / "samespk_rules.csv", index=False)
        plan = common.load_json(R / "samespk" / "plan.json")
        done = {(t["set"], t["shots"], t["index"]): t for t in tasks("samespk")}
        rows = []
        for (setname, k), group in pd.DataFrame(plan).groupby(["set", "shots"]):
            finished = [done[(setname, k, i)] for i in group["index"] if (setname, k, i) in done]
            built = [t for t in finished if t["methods"]]
            rows.append({"set": setname, "shots": k, "planned": len(group), "finished": len(finished),
                         "built": len(built), "not_built": len(finished) - len(built),
                         "enrolled_lt_k": sum(t["enrolled"] < k for t in finished),
                         "positives_built": sum(t["positives"]["clean"] for t in built),
                         "own_voice_hours_built": sum(t["negative_hours"]["own_clean"] for t in built)})
        pd.DataFrame(rows).to_csv(OUT / "samespk_enrollment.csv", index=False)

        per = pd.read_csv(R / "samespk" / "per_task.csv")
        per = per[(per.set == "main") & (per.condition == "clean")]
        built_k = per.groupby("index").shots.nunique()
        keep = set(built_k[built_k == 3].index)
        sub = per[per["index"].isin(keep) & per.rule.isin(["deployed", "bank60-tail", "global-bank", "oracle-wake"])]
        rows = []
        for (method, rule, target, k), frame in sub.groupby(["method", "rule", "target_fah", "shots"]):
            rows.append({"method": method, "rule": rule, "target_fah": target, "shots": k, "speakers": len(frame),
                         "frr": frame.frr.mean(), "fah_other_speakers": frame.fah.mean(),
                         "fah_own_pooled": float((frame.fah_own_clean * frame.hours_own_clean).sum()
                                                 / frame.hours_own_clean.sum())})
        pd.DataFrame(rows).to_csv(OUT / "samespk_paired_k.csv", index=False)
    for name, out_name, rules in (
            ("esc50", "esc50_rules.csv", ["deployed", "global-bank", "speechbank", "speechbank-os", "speechbank-tail",
                                          "speech+soundbank", "speech+soundbank-tail", "oracle-global", "oracle-wake"]),
            ("esc50_samesrc", "esc50_samesrc_rules.csv", ["deployed", "global-bank", "speechbank", "speechbank-os",
                                                          "speechbank-tail", "speech+soundbank",
                                                          "speech+soundbank-tail", "oracle-global", "oracle-wake"]),
            ("fleurs", "fleurs_rules.csv", RULES), ("condnorm", "condnorm_rules.csv", RULES)):
        path = R / name / "calibration.csv"
        if not path.is_file():
            continue
        cal = pd.read_csv(path)
        extra = [c for c in cal.columns if c.startswith(("fah_sound", "fah_speech", "fah_eval", "hours_"))]
        cal[cal.rule.isin(rules)][["shots", "method", "condition", "rule", "target_fah"] + COLS + extra] \
            .sort_values(["condition", "method", "shots", "target_fah", "rule"]).to_csv(OUT / out_name, index=False)
    xc = R / "esc50_xclass" / "calibration.csv"
    if xc.is_file():
        a = pd.read_csv(xc)
        b = pd.read_csv(R / "esc50_xclass" / "paired.csv")
        a.merge(b, on=["shots", "method", "target_fah"], how="left", suffixes=("", "_paired")) \
            .to_csv(OUT / "esc50_xclass.csv", index=False)
    cost = R / "cost"
    if (cost / "matcher_cost.csv").is_file():
        for src, dst in (("matcher_cost.csv", "cost_matcher.csv"), ("mfcc_dtw_cost.csv", "cost_mfcc_dtw.csv"),
                         ("bank_calibration_cost.csv", "cost_bank_calibration.csv")):
            pd.read_csv(cost / src).to_csv(OUT / dst, index=False)
    print("tables written to", OUT)


if __name__ == "__main__":
    main()
