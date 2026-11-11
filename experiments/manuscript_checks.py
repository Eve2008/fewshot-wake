"""Small checks for the manuscript, read from existing result files (no rescoring).

1. Orthographic variants: how many evaluation-negative and bank clips carry a label that equals an
   evaluation word after mapping Arabic letter forms to Persian ones (ك->ک, ي/ى->ی, ة->ه, أ/إ/آ->ا, ؤ->و; alef madda added after review).
2. The false alarms that remain after bank calibration (results/mswc/fa_words_seed1_k6_noaug.csv):
   how many fall on a word that contains the target, equals it after the mapping above, or is close
   in spelling (rapidfuzz ratio >= 60 on the normalised strings).

Writes results/manuscript_checks/*.csv and prints a summary.
Usage: python experiments/manuscript_checks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

OUT = common.RESULTS / "manuscript_checks"
MAP = str.maketrans({"ك": "ک", "ي": "ی", "ى": "ی", "ة": "ه", "أ": "ا", "إ": "ا", "آ": "ا", "ؤ": "و", "‌": ""})


def norm(word: str) -> str:
    return str(word).translate(MAP).strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads((common.RESULTS / "splits.json").read_text())
    targets = split["target_set"]
    tnorm = {norm(t): t for t in targets}
    table = pd.read_csv(common.MSWC / "test.csv")
    word_of = dict(zip(table.LINK, table.WORD))
    rows = []
    for seed, item in split["seeds"].items():
        words = item["negative_words"]
        hits = [w for w in words if norm(w) in tnorm and w not in targets]
        rows.append({"stream": f"eval_negatives_seed{seed}", "clips": len(words), "variant_clips": len(hits),
                     "variant_words": " ".join(sorted(set(hits)))})
    bank_words = [word_of[link] for link in split["bank"]]
    hits = [w for w in bank_words if norm(w) in tnorm and w not in targets]
    rows.append({"stream": "bank", "clips": len(bank_words), "variant_clips": len(hits),
                 "variant_words": " ".join(sorted(set(hits)))})
    variants = pd.DataFrame(rows)
    variants.to_csv(OUT / "orthographic_variants.csv", index=False)
    print(variants.to_string(index=False))

    fa = pd.read_csv(common.RESULTS / "mswc" / "fa_words_seed1_k6_noaug.csv").dropna(subset=["frame"])
    out = []
    for r in fa.itertuples():
        t, w = norm(r.target), norm(r.negative_word)
        kind = ("same word, other spelling" if w == t else "contains target" if t in w
                else "similar spelling" if fuzz.ratio(t, w) >= 60 else "other")
        out.append({"target": r.target, "negative_word": r.negative_word, "ratio": round(fuzz.ratio(t, w), 1),
                    "kind": kind})
    kinds = pd.DataFrame(out)
    kinds.to_csv(OUT / "fa_words_kinds.csv", index=False)
    summary = kinds.groupby("kind").size().rename("alarms").reset_index()
    summary.loc[len(summary)] = {"kind": "total", "alarms": len(kinds)}
    summary["wakes_with_alarms"] = [kinds[kinds.kind == k].target.nunique() if k != "total" else kinds.target.nunique()
                                    for k in summary.kind]
    summary.to_csv(OUT / "fa_words_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
