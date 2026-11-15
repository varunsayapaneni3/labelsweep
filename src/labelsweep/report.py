"""Result serialisation and Markdown report."""

from __future__ import annotations

import statistics

from .metrics import Scores
from .sweep import SweepResult, pluralize


def is_trivial_variant(wording: str, default: str) -> bool:
    """Casing or plural of the default wording (not the default itself)."""
    if wording == default:
        return False
    return wording.lower() == default.lower() or wording.lower() == (
        pluralize(default.lower()) or ""
    )


def summarize(res: dict, min_support: int = 30) -> dict:
    """Aggregate sensitivity over types with at least `min_support` gold test spans."""
    support = res.get("support", {}).get("test", {})
    used, spreads, trivial, micro = [], [], [], []
    for t, rows in res["per_type"].items():
        micro += [r["test_micro_f1"] for r in rows]
        if support.get(t, 0) < min_support:
            continue
        used.append(t)
        f = [r["test_type_f1"] for r in rows]
        spreads.append(max(f) - min(f))
        d = rows[0]
        tv = [
            r["test_type_f1"]
            for r in rows
            if is_trivial_variant(r["wording"], d["wording"])
        ]
        trivial.append(max([abs(x - d["test_type_f1"]) for x in tv], default=0.0))
    return {
        "types_used": used,
        "median_spread": statistics.median(spreads) if spreads else 0.0,
        "max_spread": max(spreads, default=0.0),
        "median_trivial_spread": statistics.median(trivial) if trivial else 0.0,
        "micro_range": (max(micro) - min(micro)) if micro else 0.0,
    }


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
        "support": {
            "dev": {t: v.tp + v.fn for t, v in r.dev_default.per_type.items()},
            "test": {t: v.tp + v.fn for t, v in r.test_default.per_type.items()},
        },
        "n_dev": r.n_dev,
        "n_test": r.n_test,
        "n_label_sets": r.n_label_sets,
        "meta": r.meta,
    }


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def markdown(res: dict, min_support: int = 30) -> str:
    m = res["meta"]
    lines = [
        f"### {m.get('data', '?')} | {m.get('model', '?')} | threshold {m.get('threshold', '?')}",
        "",
        f"dev n={res['n_dev']}, test n={res['n_test']}, label sets evaluated: {res['n_label_sets']}",
        "",
        "Per-type F1 on **test** across wordings (others held at default); best = chosen on dev.",
        "",
        "| type | test spans | default | F1 default | F1 min | F1 max | spread | dev-chosen | F1 dev-chosen (test) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    support = res.get("support", {}).get("test", {})
    for t, rows in res["per_type"].items():
        f = [r["test_type_f1"] for r in rows]
        default = rows[0]
        chosen = res["best_labels"][t]
        ch = next(r for r in rows if r["wording"] == chosen)
        n = support.get(t, "?")
        flag = "" if isinstance(n, int) and n >= min_support else " (small)"
        lines.append(
            f"| {t} | {n}{flag} | {default['wording']} | {_pct(default['test_type_f1'])} | {_pct(min(f))} | {_pct(max(f))} "
            f"| {_pct(max(f) - min(f))} | {chosen} | {_pct(ch['test_type_f1'])} |"
        )
    td, tb = res["test"]["default"], res["test"]["best"]
    lo, hi = res["test_gain_ci95"]
    sm = summarize(res, min_support)
    lines += [
        "",
        (
            f"- Types with >= {min_support} test spans: {len(sm['types_used'])}/{len(res['per_type'])}. "
            f"Median per-type spread: **{_pct(sm['median_spread'])}** F1 points (max {_pct(sm['max_spread'])}); "
            f"from casing/plural alone: **{_pct(sm['median_trivial_spread'])}**."
            if sm["types_used"]
            else f"- Median per-type spread: n/a (no type has >= {min_support} test spans; "
            "lower --min-support to see small-sample numbers, which are noisy)."
        ),
        f"- Micro-F1 range over all single-type rewordings: {_pct(sm['micro_range'])} points.",
        f"- Micro-F1 on test: default **{_pct(td['micro_f1'])}** -> dev-chosen labels **{_pct(tb['micro_f1'])}** "
        f"(gain {_pct(tb['micro_f1'] - td['micro_f1'])}, paired bootstrap 95% CI [{_pct(lo)}, {_pct(hi)}]).",
        f"- Macro-F1 on test: default {_pct(td['macro_f1'])} -> {_pct(tb['macro_f1'])}.",
        f"- Dev gain for the same choice: {_pct(res['dev']['best']['micro_f1'] - res['dev']['default']['micro_f1'])} "
        "(dev gain > test gain = selection overfit).",
    ]
    return "\n".join(lines) + "\n"


def summary_table(results: list[dict], min_support: int = 30) -> str:
    """One row per (dataset, model) result: the numbers behind H1-H4."""
    lines = [
        "| data | model | median spread | max spread | casing/plural spread | micro-F1 default | "
        "test gain (95% CI) | dev gain |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for res in sorted(
        results, key=lambda r: (r["meta"].get("data", ""), r["meta"].get("model", ""))
    ):
        sm = summarize(res, min_support)
        td, tb = res["test"]["default"]["micro_f1"], res["test"]["best"]["micro_f1"]
        dd, db = res["dev"]["default"]["micro_f1"], res["dev"]["best"]["micro_f1"]
        lo, hi = res["test_gain_ci95"]
        lines.append(
            f"| {res['meta'].get('data')} | {res['meta'].get('model', '').split('/')[-1]} "
            f"| {_pct(sm['median_spread'])} | {_pct(sm['max_spread'])} | {_pct(sm['median_trivial_spread'])} "
            f"| {_pct(td)} | {_pct(tb - td)} [{_pct(lo)}, {_pct(hi)}] | {_pct(db - dd)} |"
        )
    return "\n".join(lines) + "\n"
