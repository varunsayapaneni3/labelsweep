"""Coordinate sweep over entity-label wordings.

For each gold type, every candidate wording is tried while all other types keep their
default (first) wording. Wordings are selected on the dev split only; the spread across
wordings and the gain of the selected label set are reported on the held-out test split.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .data import Example, Span
from .metrics import Scores, counts, score


class Predictor(Protocol):
    def predict(
        self, token_lists: list[list[str]], labels: list[str]
    ) -> list[list[Span]]:
        """Return spans per example, typed with the label *wording* that was passed in."""


def validate_candidates(candidates: dict[str, list[str]]) -> None:
    owner: dict[str, str] = {}
    for typ, words in candidates.items():
        if not words:
            raise ValueError(f"type {typ!r} has no candidate wordings")
        for w in words:
            key = w.strip().lower()
            if key in owner and owner[key] != typ:
                raise ValueError(
                    f"wording {w!r} is used for both {owner[key]!r} and {typ!r}"
                )
            owner[key] = typ


def auto_variants(wording: str) -> list[str]:
    """Mechanical rewordings of a label: as-is, lowercase, Title Case, plural, '<x> name'."""
    w = wording.strip()
    low = w.lower()
    plural = low + ("es" if low.endswith(("s", "x", "ch", "sh")) else "s")
    seen: set[str] = set()
    uniq: list[str] = []
    for v in (w, low, w.title(), plural, f"{low} name"):
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


@dataclass
class WordingResult:
    wording: str
    dev_type_f1: float
    test_type_f1: float
    dev_micro_f1: float
    test_micro_f1: float


@dataclass
class SweepResult:
    defaults: dict[str, str]
    per_type: dict[str, list[WordingResult]]
    best_labels: dict[str, str]
    dev_default: Scores
    test_default: Scores
    dev_best: Scores
    test_best: Scores
    test_gain_ci: tuple[float, float]
    n_dev: int
    n_test: int
    n_label_sets: int
    meta: dict = field(default_factory=dict)

    def per_type_spread(self, typ: str, split: str = "test") -> float:
        vals = [getattr(r, f"{split}_type_f1") for r in self.per_type[typ]]
        return max(vals) - min(vals)


def per_example_counts(
    gold: Sequence[Example], pred: Sequence[Sequence[Span]], types: Sequence[str]
):
    ts = set(types)
    out = []
    for ex, ps in zip(gold, pred, strict=True):
        tp = fp = fn = 0
        for t, c in counts(ex.spans, ps).items():
            if t in ts:
                tp, fp, fn = tp + c.tp, fp + c.fp, fn + c.fn
        out.append((tp, fp, fn))
    return out


def _f1(rows) -> float:
    tp = sum(r[0] for r in rows)
    fp = sum(r[1] for r in rows)
    fn = sum(r[2] for r in rows)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def paired_bootstrap(a, b, reps: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% CI of micro-F1(b) - micro-F1(a), resampling examples with replacement."""
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(_f1([b[i] for i in idx]) - _f1([a[i] for i in idx]))
    diffs.sort()
    return diffs[int(0.025 * reps)], diffs[int(0.975 * reps) - 1]


def run_sweep(
    dev: list[Example],
    test: list[Example],
    candidates: dict[str, list[str]],
    predictor: Predictor,
    *,
    bootstrap_reps: int = 1000,
    progress: Callable[[int, tuple[str, ...]], None] | None = None,
) -> SweepResult:
    validate_candidates(candidates)
    types = list(candidates)
    defaults = {t: candidates[t][0] for t in types}
    allx = dev + test
    cache: dict[tuple[str, ...], list[list[Span]]] = {}

    def predict(assign: dict[str, str]) -> tuple[list, list]:
        labels = tuple(assign[t] for t in types)
        if labels not in cache:
            back = {assign[t]: t for t in types}
            raw = predictor.predict([ex.tokens for ex in allx], list(labels))
            cache[labels] = [
                [(s, e, back[lab]) for s, e, lab in spans if lab in back]
                for spans in raw
            ]
            if progress:
                progress(len(cache), labels)
        p = cache[labels]
        return p[: len(dev)], p[len(dev) :]

    per_type: dict[str, list[WordingResult]] = {}
    for t in types:
        rows = []
        for w in candidates[t]:
            pd, pt = predict({**defaults, t: w})
            sd, st = score(dev, pd, types), score(test, pt, types)
            rows.append(
                WordingResult(
                    w, sd.per_type[t].f1, st.per_type[t].f1, sd.micro.f1, st.micro.f1
                )
            )
        per_type[t] = rows

    # Select on dev only. max() keeps the first maximum, so ties keep the default wording.
    best = {t: max(per_type[t], key=lambda r: r.dev_type_f1).wording for t in types}
    pd0, pt0 = predict(defaults)
    pd1, pt1 = predict(best)
    ci = paired_bootstrap(
        per_example_counts(test, pt0, types),
        per_example_counts(test, pt1, types),
        reps=bootstrap_reps,
    )
    return SweepResult(
        defaults=defaults,
        per_type=per_type,
        best_labels=best,
        dev_default=score(dev, pd0, types),
        test_default=score(test, pt0, types),
        dev_best=score(dev, pd1, types),
        test_best=score(test, pt1, types),
        test_gain_ci=ci,
        n_dev=len(dev),
        n_test=len(test),
        n_label_sets=len(cache),
    )
