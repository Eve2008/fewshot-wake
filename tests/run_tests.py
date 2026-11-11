"""Minimal test runner (pytest is not installed in this environment).

Usage: python tests/run_tests.py
"""

from __future__ import annotations

import importlib
import inspect
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

failures = 0
for path in sorted(HERE.glob("test_*.py")):
    module = importlib.import_module(path.stem)
    for name, fn in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        started = time.time()
        try:
            if "tmp_path" in inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as tmp:
                    fn(Path(tmp))
            else:
                fn()
            print(f"PASS {path.stem}.{name} ({time.time() - started:.1f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {path.stem}.{name}: {exc!r}", flush=True)
sys.exit(1 if failures else 0)
