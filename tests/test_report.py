from labelsweep.report import is_trivial_variant, summarize


def test_trivial_variants_are_casing_and_plural_only():
    assert is_trivial_variant("Organization", "organization")
    assert is_trivial_variant("countries", "country")
    assert not is_trivial_variant("nation", "country")
    assert not is_trivial_variant(
        "country", "country"
    )  # the default itself is not a variant


def _res(rows_by_type, support):
    return {
        "per_type": {
            t: [
                {
                    "wording": w,
                    "test_type_f1": f,
                    "dev_type_f1": f,
                    "dev_micro_f1": 0,
                    "test_micro_f1": m,
                }
                for w, f, m in rows
            ]
            for t, rows in rows_by_type.items()
        },
        "support": {"test": support},
    }


def test_summarize_separates_trivial_spread_and_filters_small_types():
    res = _res(
        {
            "country": [
                ("country", 0.8, 0.5),
                ("nation", 0.4, 0.45),
                ("Country", 0.7, 0.49),
            ],
            "rare": [("rare", 0.0, 0.5), ("Rare", 1.0, 0.6)],
        },
        {"country": 100, "rare": 3},
    )
    s = summarize(res, min_support=30)
    assert s["types_used"] == ["country"]
    assert abs(s["median_spread"] - 0.4) < 1e-9
    assert abs(s["median_trivial_spread"] - 0.1) < 1e-9
    assert (
        abs(s["micro_range"] - 0.15) < 1e-9
    )  # micro range spans all types, small ones included


def test_summary_table_has_one_row_per_result():
    from labelsweep.report import summary_table

    base = _res(
        {"country": [("country", 0.8, 0.5), ("nation", 0.4, 0.45)]}, {"country": 100}
    )
    for name, model in (("crossner_ai", "m/small"), ("crossner_ai", "m/large")):
        base_i = dict(base)
        base_i["meta"] = {"data": name, "model": model}
        base_i["test"] = {"default": {"micro_f1": 0.5}, "best": {"micro_f1": 0.52}}
        base_i["dev"] = {"default": {"micro_f1": 0.5}, "best": {"micro_f1": 0.6}}
        base_i["test_gain_ci95"] = [-0.01, 0.05]
        if name == "crossner_ai" and model == "m/small":
            first = base_i
        else:
            second = base_i
    table = summary_table([first, second])
    assert table.count("| crossner_ai |") == 2
    assert "| large |" in table and "40.0" in table
