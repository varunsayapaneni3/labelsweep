"""Result serialisation and Markdown report."""

from __future__ import annotations

import statistics

from .metrics import Scores
from .sweep import SweepResult


def _scores(s: Scores) -> dict:
    return {
        "micro_p": s.micro.precision,
        "micro_r": s.micro.recall,
        "micro_f1": s.micro.f1,
        "macro_f1": s.macro_f1,
        "per_type_f1": {t: v.f1 for t, v in s.per_type.items()},
    }


def to_dict(r: SweepResult) -> dict:
    return {
        "defaults": r.defaults,
        "best_labels": r.best_labels,
        "per_type": {
            t: [
                {
                    "wording": w.wording,
                    "dev_type_f1": w.dev_type_f1,
                    "test_type_f1": w.test_type_f1,
                    "dev_micro_f1": w.dev_micro_f1,
                    "test_micro_f1": w.test_micro_f1,
                }
                for w in rows
            ]
            for t, rows in r.per_type.items()
        },
        "dev": {"default": _scores(r.dev_default), "best": _scores(r.dev_best)},
        "test": {"default": _scores(r.test_default), "best": _scores(r.test_best)},
        "test_gain_ci95": list(r.test_gain_ci),
        "n_dev": r.n_dev,
        "n_test": r.n_test,
        "n_label_sets": r.n_label_sets,
        "meta": r.meta,
    }


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def markdown(res: dict) -> str:
    m = res["meta"]
    lines = [
        f"### {m.get('data', '?')} | {m.get('model', '?')} | threshold {m.get('threshold', '?')}",
        "",
        f"dev n={res['n_dev']}, test n={res['n_test']}, label sets evaluated: {res['n_label_sets']}",
        "",
        "Per-type F1 on **test** across wordings (others held at default); best = chosen on dev.",
        "",
        "| type | default | F1 default | F1 min | F1 max | spread | dev-chosen | F1 dev-chosen (test) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    spreads = []
    for t, rows in res["per_type"].items():
        f = [r["test_type_f1"] for r in rows]
        default = rows[0]
        chosen = res["best_labels"][t]
        ch = next(r for r in rows if r["wording"] == chosen)
        spreads.append(max(f) - min(f))
        lines.append(
            f"| {t} | {default['wording']} | {_pct(default['test_type_f1'])} | {_pct(min(f))} | {_pct(max(f))} "
            f"| {_pct(max(f) - min(f))} | {chosen} | {_pct(ch['test_type_f1'])} |"
        )
    td, tb = res["test"]["default"], res["test"]["best"]
    lo, hi = res["test_gain_ci95"]
    lines += [
        "",
        f"- Median per-type spread: **{_pct(statistics.median(spreads))}** F1 points "
        f"(max {_pct(max(spreads))}).",
        f"- Micro-F1 on test: default **{_pct(td['micro_f1'])}** -> dev-chosen labels **{_pct(tb['micro_f1'])}** "
        f"(gain {_pct(tb['micro_f1'] - td['micro_f1'])}, paired bootstrap 95% CI [{_pct(lo)}, {_pct(hi)}]).",
        f"- Macro-F1 on test: default {_pct(td['macro_f1'])} -> {_pct(tb['macro_f1'])}.",
        f"- Dev gain for the same choice: {_pct(res['dev']['best']['micro_f1'] - res['dev']['default']['micro_f1'])} "
        "(dev gain > test gain = selection overfit).",
    ]
    return "\n".join(lines) + "\n"
