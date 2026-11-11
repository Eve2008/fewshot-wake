"""Size and single-thread cost of the mHuBERT-147 front end, offline and with limited context (added after review).

Counts the parameters of the 6-layer model used by run_ssl.py / run_ssl_lc.py (fp32 and fp16 sizes), and times
with one PyTorch thread (thread CPU time, 3 repeats, warm-up first):
  offline   one forward pass over 10 s of audio plus 1.28 s of context on each side (as cost_ssl.py)
  limited   one forward pass over the 1.285-s window that run_ssl_lc.py uses for every 0.32 s block
Both are reported as ms of computation per 80 ms of audio. The machine runs other jobs; the load average is saved.

Output: results/cost/ssl_lc_cost.json
Usage: python experiments/cost_ssl_lc.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["SSL_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import common  # noqa: E402
import run_ssl  # noqa: E402
import run_ssl_lc  # noqa: E402


def timed(model, x, repeats: int = 3) -> list[float]:
    import torch

    out = []
    with torch.inference_mode():
        model(x, output_hidden_states=True)                     # warm-up
        for _ in range(repeats):
            start = time.thread_time()
            model(x, output_hidden_states=True)
            out.append(time.thread_time() - start)
    return out


def main() -> None:
    import torch

    torch.set_num_threads(1)
    layer = run_ssl.chosen_layer()
    model = run_ssl.model(layer)
    params = int(sum(p.numel() for p in model.parameters()))
    rng = np.random.default_rng(0)
    sr = common.SR
    seconds, context = 10.0, run_ssl.CONTEXT / sr
    x_off = torch.from_numpy(rng.normal(0, 1, int((seconds + 2 * context) * sr)).astype(np.float32))[None]
    t_off = timed(model, x_off)
    window = run_ssl_lc.LEFT + run_ssl_lc.BLOCK + run_ssl_lc.TAIL
    x_lc = torch.from_numpy(rng.normal(0, 1, window).astype(np.float32))[None]
    t_lc = timed(model, x_lc, repeats=20)
    block_frames = run_ssl_lc.BLOCK / sr / 0.08                  # 80-ms frames per model call
    off = [t / (seconds / 0.08) * 1000 for t in t_off]
    lc = [t / block_frames * 1000 for t in t_lc]
    out = {"layers": layer, "parameters": params, "size_mb_fp32": params * 4 / 1e6, "size_mb_fp16": params * 2 / 1e6,
           "threads": 1, "offline": {"audio_seconds": seconds, "context_seconds_each_side": context,
                                     "thread_cpu_seconds": t_off, "ms_per_80ms_median": float(np.median(off)),
                                     "ms_per_80ms_range": [float(min(off)), float(max(off))]},
           "limited_context": {"window_samples": int(window), "block_seconds": run_ssl_lc.BLOCK / sr,
                               "thread_cpu_seconds": t_lc, "ms_per_80ms_median": float(np.median(lc)),
                               "ms_per_80ms_range": [float(min(lc)), float(max(lc))]},
           "load_average": os.getloadavg()}
    (common.RESULTS / "cost").mkdir(parents=True, exist_ok=True)
    (common.RESULTS / "cost" / "ssl_lc_cost.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
