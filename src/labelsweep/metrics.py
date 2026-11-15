"""Strict span-level precision/recall/F1 (exact boundaries and type)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .data import Example, Span


@dataclass
class PRF:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class Scores:
    micro: PRF
    per_type: dict[str, PRF]

    @property
    def macro_f1(self) -> float:
        return (
            sum(v.f1 for v in self.per_type.values()) / len(self.per_type)
            if self.per_type
            else 0.0
        )


def counts(gold: Sequence[Span], pred: Sequence[Span]) -> dict[str, PRF]:
    g, p = set(gold), set(pred)
    out: dict[str, PRF] = {}
    for s in g & p:
        out.setdefault(s[2], PRF()).tp += 1
    for s in p - g:
        out.setdefault(s[2], PRF()).fp += 1
    for s in g - p:
        out.setdefault(s[2], PRF()).fn += 1
    return out


def score(
    gold: Sequence[Example], pred: Sequence[Sequence[Span]], types: Sequence[str]
) -> Scores:
    per = {t: PRF() for t in types}
    for ex, ps in zip(gold, pred, strict=True):
        for t, c in counts(ex.spans, ps).items():
            if t in per:
                per[t].tp += c.tp
                per[t].fp += c.fp
                per[t].fn += c.fn
    micro = PRF(
        sum(v.tp for v in per.values()),
        sum(v.fp for v in per.values()),
        sum(v.fn for v in per.values()),
    )
    return Scores(micro, per)
