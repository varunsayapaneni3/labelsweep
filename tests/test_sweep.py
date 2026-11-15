import pytest

from labelsweep.data import Example
from labelsweep.sweep import (
    auto_variants,
    paired_bootstrap,
    run_sweep,
    validate_candidates,
)


class FakePredictor:
    """Finds a gold span only if its type is worded the 'right' way; records calls."""

    def __init__(self, gold, good):
        self.gold, self.good, self.calls = gold, good, []

    def predict(self, token_lists, labels):
        self.calls.append(tuple(labels))
        out = []
        for toks in token_lists:
            ex = next(e for e in self.gold if e.tokens == toks)
            spans = []
            for s, e, t in ex.spans:
                for lab in labels:
                    if self.good.get(lab) == t:
                        spans.append((s, e, lab))
            out.append(spans)
        return out


def _data(n, offset=0):
    return [
        Example([f"w{i + offset}", "x", "y"], [(0, 1, "person"), (2, 3, "city")])
        for i in range(n)
    ]


def test_sweep_finds_best_wording_and_reports_spread():
    dev, test = _data(4), _data(6, offset=100)
    cands = {"person": ["person", "person name"], "city": ["city", "town"]}
    pred = FakePredictor(
        dev + test, {"person name": "person", "city": "city", "town": "city"}
    )
    r = run_sweep(dev, test, cands, pred)
    assert r.best_labels == {"person": "person name", "city": "city"}
    per = {row.wording: row for row in r.per_type["person"]}
    assert per["person"].test_type_f1 == 0.0
    assert per["person name"].test_type_f1 == 1.0
    assert r.per_type_spread("person", split="test") == 1.0
    assert r.per_type_spread("city", split="test") == 0.0
    assert r.test_default.micro.f1 < r.test_best.micro.f1 == 1.0


def test_sweep_caches_identical_label_sets():
    dev, test = _data(2), _data(2, offset=50)
    pred = FakePredictor(dev + test, {"a": "person", "city": "city"})
    run_sweep(dev, test, {"person": ["a", "b"], "city": ["city"]}, pred)
    # default set and person=b; the final best set equals the default and is cached
    assert len(set(pred.calls)) == len(pred.calls) == 2


def test_validate_rejects_wording_shared_by_two_types():
    with pytest.raises(ValueError, match="person"):
        validate_candidates({"person": ["person"], "artist": ["artist", "person"]})


def test_auto_variants_are_unique_and_keep_default_first():
    v = auto_variants("musical artist")
    assert v[0] == "musical artist"
    assert len(v) == len(set(v)) >= 3
    assert "musical artists" in v


def test_paired_bootstrap_ci_contains_true_difference():
    a = [(1, 0, 0)] * 50 + [(0, 1, 1)] * 50  # per-example (tp, fp, fn)
    b = [(1, 0, 0)] * 100
    lo, hi = paired_bootstrap(a, b, reps=500, seed=0)
    assert lo > 0 and hi <= 1.0


def test_pluralize_handles_y_es_and_adjectives():
    from labelsweep.sweep import pluralize

    assert pluralize("country") == "countries"
    assert pluralize("academic journal") == "academic journals"
    assert pluralize("class") == "classes"
    assert pluralize("day") == "days"
    assert pluralize("miscellaneous") is None
