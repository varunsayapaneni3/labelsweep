"""Run the pre-registered grid: every bundled CrossNER/MIT preset x GLiNER v2.1 sizes.

Skips result files that already exist. Usage: python scripts/run_grid.py [--models ...] [--data ...]
"""

import argparse
import subprocess
import sys
from pathlib import Path

DATA = [
    "crossner_ai",
    "crossner_literature",
    "crossner_music",
    "crossner_politics",
    "crossner_science",
    "mit_restaurant",
]
MODELS = [
    "urchade/gliner_small-v2.1",
    "urchade/gliner_medium-v2.1",
    "urchade/gliner_large-v2.1",
]

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=MODELS)
ap.add_argument("--data", nargs="+", default=DATA)
ap.add_argument("--outdir", default="results")
a = ap.parse_args()
for m in a.models:
    for d in a.data:
        out = Path(a.outdir) / f"{d}__{m.replace('/', '__')}.json"
        if out.exists():
            print("skip", out, flush=True)
            continue
        print("RUN", d, m, flush=True)
        rc = subprocess.call(
            [
                sys.executable,
                "-m",
                "labelsweep",
                "run",
                d,
                "--model",
                m,
                "--out",
                str(out),
            ],
            stdout=subprocess.DEVNULL,
        )
        print("done", d, m, "rc", rc, flush=True)
