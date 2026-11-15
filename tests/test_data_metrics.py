from labelsweep.data import Example, bio_to_spans
from labelsweep.metrics import score


def test_bio_to_spans_basic():
    tags = ["B-person", "I-person", "O", "B-country", "O"]
    assert bio_to_spans(tags) == [(0, 2, "person"), (3, 4, "country")]


def test_bio_to_spans_stray_inside_starts_new_span():
    # an I- tag without a matching B- starts a span (conlleval convention)
    assert bio_to_spans(["O", "I-band", "I-band"]) == [(1, 3, "band")]


def test_bio_to_spans_type_change_breaks_span():
    assert bio_to_spans(["B-a", "I-b"]) == [(0, 1, "a"), (1, 2, "b")]


def test_score_strict_micro_and_per_type():
    gold = [Example(["x"] * 6, [(0, 2, "person"), (3, 4, "country")])]
    pred = [[(0, 2, "person"), (3, 5, "country"), (5, 6, "person")]]
    s = score(gold, pred, types=["person", "country"])
    # tp=1 (person), fp=2, fn=1
    assert round(s.micro.precision, 4) == round(1 / 3, 4)
    assert s.micro.recall == 0.5
    assert (
        s.per_type["person"].f1 == round(2 * 0.5 * 1 / 1.5, 10)
        or abs(s.per_type["person"].f1 - 2 / 3) < 1e-9
    )
    assert s.per_type["country"].f1 == 0.0
    assert abs(s.macro_f1 - (2 / 3 + 0) / 2) < 1e-9


def test_score_empty_is_zero_not_error():
    s = score([Example(["a"], [])], [[]], types=["x"])
    assert s.micro.f1 == 0.0
