from labelsweep.predictors import char_to_token_spans, join_tokens


def test_join_tokens_offsets():
    text, starts, ends = join_tokens(["Geoffrey", "Hinton", "."])
    assert text == "Geoffrey Hinton ."
    assert starts == [0, 9, 16] and ends == [8, 15, 17]


def test_exact_char_spans_map_to_tokens():
    toks = ["Geoffrey", "Hinton", "met", "LeCun"]
    ents = [
        {"start": 0, "end": 15, "label": "person"},
        {"start": 20, "end": 25, "label": "person"},
    ]
    assert char_to_token_spans(toks, ents) == [(0, 2, "person"), (3, 4, "person")]


def test_mid_token_offsets_snap_to_containing_tokens():
    toks = ["U.S.", "Army"]
    # model split "U.S." and returned only "S. Army"
    ents = [{"start": 2, "end": 9, "label": "org"}]
    assert char_to_token_spans(toks, ents) == [(0, 2, "org")]


def test_duplicate_spans_after_snapping_are_dropped():
    toks = ["ab"]
    ents = [{"start": 0, "end": 1, "label": "x"}, {"start": 1, "end": 2, "label": "x"}]
    assert char_to_token_spans(toks, ents) == [(0, 1, "x")]
