"""labelsweep command line."""

from __future__ import annotations

import argparse
import glob
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .config import list_presets, load_labels
from .data import BENCHMARKS, load_hf, load_jsonl
from .report import markdown, summary_table, to_dict
from .sweep import run_sweep


def make_predictor(args):
    from .predictors import GlinerPredictor

    return GlinerPredictor(
        args.model, threshold=args.threshold, batch_size=args.batch_size
    )


def _load_data(spec: str, dev_limit: int | None, test_limit: int | None):
    if "," in spec:
        dev_path, test_path = spec.split(",", 1)
        dev, test = load_jsonl(dev_path), load_jsonl(test_path)
    else:
        dev, test = load_hf(spec, "validation"), load_hf(spec, "test")
    return dev[:dev_limit] if dev_limit else dev, test[
        :test_limit
    ] if test_limit else test


def _versions() -> dict:
    out = {"labelsweep": __version__, "python": platform.python_version()}
    for mod in ("gliner", "torch", "transformers"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    return out


def _err(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


def expand_paths(patterns: list[str]) -> list[str]:
    """Expand globs and directories ourselves: PowerShell and cmd pass `*.json` through literally."""
    out: list[str] = []
    for p in patterns:
        if Path(p).is_dir():
            out += sorted(str(x) for x in Path(p).glob("*.json"))
        elif glob.has_magic(p):
            out += sorted(glob.glob(p))
        elif Path(p).exists():
            out.append(p)
    return out


def _default_out(args) -> Path:
    stem = Path(args.data.split(",")[0]).stem if "," in args.data else args.data
    return Path(f"results/{stem}__{args.model.replace('/', '__')}.json")


def cmd_run(args) -> int:
    labels_spec = args.labels or (
        f"preset:{args.data}" if args.data in list_presets() else None
    )
    if not labels_spec:
        return _err(
            "--labels is required (no bundled preset for this data); see examples/labels.yaml"
        )
    if not labels_spec.startswith("preset:") and not Path(labels_spec).is_file():
        return _err(
            f"labels file not found: {labels_spec} (try examples/labels.yaml or preset:<name>)"
        )
    if "," in args.data:
        for p in args.data.split(",", 1):
            if not Path(p).is_file():
                return _err(
                    f"data file not found: {p} (expected DEV.jsonl,TEST.jsonl; try examples/)"
                )
    elif args.data not in BENCHMARKS:
        return _err(
            f"unknown dataset {args.data!r}; use one of {', '.join(BENCHMARKS)} or DEV.jsonl,TEST.jsonl"
        )
    out = Path(args.out) if args.out else _default_out(args)
    if out.exists() and not args.force:
        return _err(
            f"{out} exists; pass --force to overwrite or --out to write elsewhere"
        )
    candidates = load_labels(labels_spec, auto=args.auto)
    dev, test = _load_data(args.data, args.dev_limit, args.test_limit)
    n_sets = 1 + sum(len(v) - 1 for v in candidates.values())
    print(
        f"{args.data}: dev={len(dev)} test={len(test)} types={len(candidates)} label sets<= {n_sets + 1}",
        file=sys.stderr,
    )
    predictor = make_predictor(args)
    t0 = time.time()

    def progress(i, labels):
        print(
            f"  [{i}/{n_sets + 1}] {time.time() - t0:6.1f}s",
            file=sys.stderr,
            flush=True,
        )

    r = run_sweep(
        dev,
        test,
        candidates,
        predictor,
        bootstrap_reps=args.bootstrap,
        progress=progress,
    )
    r.meta = {
        "data": args.data,
        "labels": labels_spec,
        "auto": args.auto,
        "model": args.model,
        "threshold": args.threshold,
        "seconds": round(time.time() - t0, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": _versions(),
    }
    res = to_dict(r)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(markdown(res))
    print(f"wrote {out}", file=sys.stderr)
    return 0


def cmd_report(args) -> int:
    paths = expand_paths(args.results)
    if not paths:
        return _err(f"no result files match {' '.join(args.results)}")
    results = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    if args.summary:
        print(summary_table(results, args.min_support))
        return 0
    for res in results:
        print(markdown(res, args.min_support))
    return 0


def cmd_presets(args) -> int:
    for n in list_presets():
        src = BENCHMARKS.get(n)
        print(f"{n}\t{src[0] + ':' + src[1] if src else ''}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="labelsweep", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="sweep label wordings on a dataset")
    r.add_argument(
        "data", help="benchmark name (see `labelsweep presets`) or DEV.jsonl,TEST.jsonl"
    )
    r.add_argument(
        "--labels", help="YAML of type -> [default, alternatives...] or preset:<name>"
    )
    r.add_argument(
        "--auto", action="store_true", help="expand types that list a single wording"
    )
    r.add_argument("--model", default="urchade/gliner_small-v2.1")
    r.add_argument("--threshold", type=float, default=0.5)
    r.add_argument("--batch-size", type=int, default=16)
    r.add_argument("--dev-limit", type=int)
    r.add_argument("--test-limit", type=int)
    r.add_argument("--bootstrap", type=int, default=1000)
    r.add_argument("--out")
    r.add_argument(
        "--force", action="store_true", help="overwrite an existing result file"
    )
    r.set_defaults(fn=cmd_run)

    p = sub.add_parser("report", help="print Markdown tables for result JSON files")
    p.add_argument(
        "results",
        nargs="+",
        help="result JSON files, globs (results/*.json) or a directory",
    )
    p.add_argument("--summary", action="store_true", help="one row per result file")
    p.add_argument(
        "--min-support",
        type=int,
        default=30,
        help="min gold test spans for a type to count",
    )
    p.set_defaults(fn=cmd_report)

    s = sub.add_parser("presets", help="list bundled label presets")
    s.set_defaults(fn=cmd_presets)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
