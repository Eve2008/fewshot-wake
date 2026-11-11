"""Figures for the paper (PNG at 200 dpi + PDF) in results/figures/.

Usage: python experiments/figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

R = common.RESULTS
FIG = R / "figures"
# Reference categorical palette (fixed order) from the dataviz guidance, light mode.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LABEL = {"nexa": "Deployed (noise-augmented templates)", "nexa-noaug": "Clean templates",
         "nexa-noaug-bnorm": "Clean templates, bank normaliser", "nexa-nonorm": "No normaliser",
         "nexa-min": "Best single template", "emb-only": "Embedding only", "mel-only": "Log-mel only",
         "prototype": "Prototype", "mfcc-dtw": "MFCC + DTW"}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.bbox": "tight", "font.family": "DejaVu Sans"})


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def det_figure() -> None:
    det = pd.read_csv(R / "mswc" / "det.csv")
    cal = pd.read_csv(R / "mswc" / "calibration.csv")
    methods = ["nexa", "nexa-noaug", "emb-only", "mfcc-dtw", "prototype", "nexa-nonorm"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=True)
    for ax, cond in zip(axes, ("clean", "noisy")):
        for i, m in enumerate(methods):
            d = det[(det.shots == 6) & (det.method == m) & (det.condition == cond)].sort_values("tau")
            d = d[d.fah > 0]
            ax.plot(d.fah, d.frr, color=C[i], lw=1.6, label=LABEL[m])
        for i, m in enumerate(("nexa", "nexa-noaug")):
            for rule, mk in (("deployed", "o"), ("bank60-tail", "s")):
                a = cal[(cal.shots == 6) & (cal.method == m) & (cal.condition == cond) & (cal.rule == rule)
                        & (cal.target_fah == 1.0)]
                if not a.empty:
                    ax.plot(max(a.fah_mean.iloc[0], 0.012), a.frr.iloc[0], mk, ms=7, mfc="white", mec=C[i], mew=1.6)
        ax.set_xscale("log")
        ax.set_xlim(0.01, 3000)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("False alarms per hour (mean over wakes)")
        ax.set_title(f"{cond.capitalize()} negatives and positives", fontsize=9, color=INK)
    axes[0].set_ylabel("False-reject rate (mean over wakes)")
    axes[0].plot([], [], "o", mfc="white", mec=INK2, label="deployed calibration")
    axes[0].plot([], [], "s", mfc="white", mec=INK2, label="bank calibration, 1 FA/h target")
    fig.subplots_adjust(bottom=0.33, wspace=0.08)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig_det_k6")


def calibration_figure() -> None:
    per = pd.read_csv(R / "mswc" / "per_task.csv")
    rules = ["deployed", "global-bank", "bank15", "bank30", "bank60", "bank60-os", "bank60-tail", "oracle-wake"]
    names = ["Deployed\n(20 s room)", "Global\n(bank)", "Bank\n15 min", "Bank\n30 min", "Bank\n60 min",
             "Bank 60\norder stat.", "Bank 60\ntail fit", "Oracle\nper wake"]
    sel = per[(per.method == "nexa-noaug") & (per.condition == "clean") & (per.target_fah == 1.0)]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    rng = np.random.default_rng(0)
    floor = 0.05
    for i, rule in enumerate(rules):
        v = sel[sel.rule == rule].fah.to_numpy()
        y = np.maximum(v, floor)
        x = i + rng.uniform(-0.18, 0.18, size=y.size)
        ax.scatter(x, y, s=10, color=C[0], alpha=0.55, linewidths=0)
        ax.plot([i - 0.3, i + 0.3], [max(np.median(v), floor)] * 2, color=INK, lw=1.6)
        ax.plot(i, max(np.mean(v), floor), "D", color=C[1], ms=5)
    ax.axhline(1.0, color=INK2, lw=0.9, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(floor * 0.8, 1000)
    ax.set_xticks(range(len(rules)), names, fontsize=7.5)
    ax.set_ylabel("Achieved false alarms per hour\n(evaluation negatives)")
    ax.plot([], [], "-", color=INK, label="median")
    ax.plot([], [], "D", color=C[1], label="mean")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.text(len(rules) - 0.5, 1.15, "target 1 FA/h", fontsize=7, color=INK2, ha="right")
    ax.set_yticks([floor, 0.1, 1, 10, 100, 1000], ["0", "0.1", "1", "10", "100", "1000"])
    ax.set_title("Clean templates, clean condition, k = 3, 6 and 10 pooled (144 wakes)", fontsize=9, color=INK)
    save(fig, "fig_calibration")


def shots_figure() -> None:
    cal = pd.read_csv(R / "mswc" / "calibration.csv")
    same = pd.read_csv(R / "samespk" / "calibration.csv") if (R / "samespk" / "calibration.csv").is_file() else None
    methods = ["nexa", "nexa-noaug", "emb-only", "mfcc-dtw", "prototype"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    for ax, (title, frame, extra) in zip(axes, (("Cross-speaker (16 words x 3 seeds)", cal, {}),
                                                ("Same speaker (main set)", same, {"set": "main"}))):
        if frame is None:
            continue
        for i, m in enumerate(methods):
            a = frame[(frame.method == m) & (frame.condition == "clean") & (frame.rule == "bank60-tail")
                      & (frame.target_fah == 1.0)]
            for key, value in extra.items():
                a = a[a[key] == value]
            a = a.sort_values("shots")
            if a.empty:
                continue
            ax.errorbar(a.shots + (i - 2) * 0.12, a.frr, yerr=[a.frr - a.frr_lo, a.frr_hi - a.frr], color=C[i],
                        marker=MARKERS[i], ms=5, lw=1.4, capsize=2, label=LABEL[m])
            # Same speaker: the FRR of a method means little without its false alarms on the speaker's own
            # other words (MFCC + DTW responds to the voice), so those are printed next to the points.
            if extra and m in ("nexa-noaug", "mfcc-dtw") and "fah_own_clean_pooled" in a:
                for x, y, own in zip(a.shots + (i - 2) * 0.12, a.frr, a.fah_own_clean_pooled):
                    ax.annotate(f"{own:.1f}", (x, y), xytext=(7, -11 if m == "mfcc-dtw" else 8),
                                textcoords="offset points", fontsize=6.5, color=C[i])
        if extra:
            ax.text(0.02, 0.03, "Numbers: FA/h on the same speaker's other words", transform=ax.transAxes,
                    fontsize=6.5, color=INK2)
        ax.set_xticks([3, 6, 10])
        ax.set_xlabel("Enrollment recordings k")
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("FRR at the bank-calibrated threshold\n(target 1 FA/h, tail fit)")
    fig.subplots_adjust(bottom=0.33, wspace=0.08)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig_shots")


def esc_figure() -> None:
    path = R / "esc50" / "per_class_noaug_k6.csv"
    if not path.is_file():
        return
    cls = pd.read_csv(path).sort_values("frr", ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 6.8))
    y = np.arange(len(cls))
    ax.barh(y, 1 - cls.frr, color=C[0], height=0.7)
    ax.set_yticks(y, [c.replace("_", " ") for c in cls.category], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Detection rate (1 - FRR), clean templates, k = 6, speech + sound bank, 1 FA/h target")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_esc50_classes")


def cost_figure() -> None:
    path = R / "cost" / "matcher_cost.csv"
    if not path.is_file():
        return
    c = pd.read_csv(path)
    info = common.load_json(R / "cost" / "cost_info.json")
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for i, length in enumerate(sorted(c.length.unique())):
        a = c[c.length == length].sort_values("templates")
        ax.errorbar(a.templates, a.detector_ms_median,
                    yerr=[a.detector_ms_median - a.detector_ms_median_min,
                          a.detector_ms_median_max - a.detector_ms_median],
                    marker=MARKERS[i], color=C[i], lw=1.6, ms=4, capsize=2,
                    label=f"deployed NumPy matcher, {length}-frame templates")
        ax.plot(a.templates, a.compiled_ms, marker=MARKERS[i], color=C[i], lw=1.0, ls="--", ms=4, mfc="white",
                label=f"compiled matcher, {length}-frame templates")
    feat = info["features"]["feature_ms_median"]
    ax.axhline(feat, color=INK2, lw=0.9, ls=":")
    ax.text(2.6, feat * 1.18, f"feature extraction ({feat:.2f} ms)", fontsize=7, color=INK2)
    budget = common.FRAME_SECONDS * 1000
    ax.axhline(budget, color=C[7], lw=0.9, ls="-.")
    ax.text(2.6, budget * 1.18, "real-time budget (80 ms per frame)", fontsize=7, color=C[7])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([3, 6, 10, 18, 30, 60, 120, 240], ["3", "6", "10", "18", "30", "60", "120", "240"])
    ax.minorticks_off()
    ax.set_xlabel("Templates in the matcher")
    ax.set_ylabel("ms per 80 ms frame (single thread)")
    ax.set_ylim(top=budget * 3)
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda j: (not labels[j].startswith("deployed"), int(labels[j].split(", ")[1].split("-")[0])))
    ax.legend([handles[j] for j in order], [labels[j] for j in order], fontsize=6.5, loc="center left",
              bbox_to_anchor=(1.01, 0.5))
    save(fig, "fig_cost")


if __name__ == "__main__":
    for fn in (det_figure, calibration_figure, shots_figure, esc_figure, cost_figure):
        try:
            fn()
            print("ok", fn.__name__)
        except Exception as exc:  # noqa: BLE001
            print("failed", fn.__name__, repr(exc))
