"""Single-thread cost of the mHuBERT-147 (6 transformer layers) features used by the SSL baseline.

Times the forward pass over 10 s of audio with 1.28 s of context on each side (the offline windows of
run_ssl.py use 30 s), one PyTorch thread, thread CPU time, 3 repeats, on whatever else the machine is running.
Reports ms of computation per 80 ms of audio, comparable with cost.py's feature extraction.

Output: results/cost/ssl_cost.json
Usage: python experiments/cost_ssl.py
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


def main() -> None:
    import torch

    torch.set_num_threads(1)
    layer = run_ssl.chosen_layer()
    model = run_ssl.model(layer)
    rng = np.random.default_rng(0)
    seconds, context = 10.0, run_ssl.CONTEXT / common.SR
    x = torch.from_numpy(rng.normal(0, 1, int((seconds + 2 * context) * common.SR)).astype(np.float32))[None]
    times = []
    with torch.inference_mode():
        model(x, output_hidden_states=True)                     # warm-up
        for _ in range(3):
            start = time.thread_time()
            model(x, output_hidden_states=True)
            times.append(time.thread_time() - start)
    per_frame = [t / (seconds / 0.08) * 1000 for t in times]
    out = {"layers": layer, "audio_seconds": seconds, "context_seconds_each_side": context, "threads": 1,
           "thread_cpu_seconds": times, "ms_per_80ms_frame_median": float(np.median(per_frame)),
           "ms_per_80ms_frame_range": [float(min(per_frame)), float(max(per_frame))],
           "load_average": os.getloadavg()}
    (common.RESULTS / "cost").mkdir(parents=True, exist_ok=True)
    (common.RESULTS / "cost" / "ssl_cost.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
