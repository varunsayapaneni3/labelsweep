import json

import pytest

from labelsweep import cli
from labelsweep.config import list_presets, load_labels
from labelsweep.sweep import validate_candidates


class Perfect:
    """Predicts gold spans whenever the type's wording is in `good`."""

    def __init__(self, gold, good):
        self.by_tokens = {tuple(e["tokens"]): e["spans"] for e in gold}
        self.good = good

    def predict(self, token_lists, labels):
        out = []
        for toks in token_lists:
            spans = []
            for s, e, t in self.by_tokens[tuple(toks)]:
                for lab in labels:
                    if self.good.get(lab) == t:
                        spans.append((s, e, lab))
            out.append(spans)
        return out


@pytest.fixture
def files(tmp_path):
    dev = [{"tokens": [f"d{i}", "Paris"], "spans": [[1, 2, "city"]]} for i in range(5)]
    test = [{"tokens": [f"t{i}", "Rome"], "spans": [[1, 2, "city"]]} for i in range(8)]
    for name, rows in (("dev", dev), ("test", test)):
        (tmp_path / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )
    (tmp_path / "labels.yaml").write_text("city: [town, city]\n", encoding="utf-8")
    return tmp_path, dev + test


def test_run_writes_result_json_and_report_prints_table(files, monkeypatch, capsys):
    d, gold = files
    monkeypatch.setattr(
        cli, "make_predictor", lambda args: Perfect(gold, {"city": "city"})
    )
    out = d / "res.json"
    rc = cli.main(
        [
            "run",
            f"{d / 'dev.jsonl'},{d / 'test.jsonl'}",
            "--labels",
            str(d / "labels.yaml"),
            "--model",
            "fake",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["best_labels"] == {"city": "city"}
    assert res["test"]["default"]["micro_f1"] == 0.0
    assert res["test"]["best"]["micro_f1"] == 1.0
    assert res["n_test"] == 8 and res["meta"]["model"] == "fake"
    capsys.readouterr()
    assert cli.main(["report", str(out)]) == 0
    text = capsys.readouterr().out
    assert "| city |" in text and "town" in text


def test_auto_expands_single_wording_types(tmp_path):
    p = tmp_path / "l.yaml"
    p.write_text("person: [person]\norg: [organization, company]\n", encoding="utf-8")
    c = load_labels(str(p), auto=True)
    assert c["person"][0] == "person" and len(c["person"]) > 1
    assert c["org"] == ["organization", "company"]


def test_every_bundled_preset_is_valid():
    names = list_presets()
    assert "crossner_ai" in names
    for n in names:
        validate_candidates(load_labels(f"preset:{n}"))


def test_report_expands_globs_itself_for_powershell(files, monkeypatch, capsys):
    d, gold = files
    monkeypatch.setattr(
        cli, "make_predictor", lambda args: Perfect(gold, {"city": "city"})
    )
    for name in ("a", "b"):
        cli.main(
            [
                "run",
                f"{d / 'dev.jsonl'},{d / 'test.jsonl'}",
                "--labels",
                str(d / "labels.yaml"),
                "--model",
                "fake",
                "--out",
                str(d / "res" / f"{name}.json"),
            ]
        )
    capsys.readouterr()
    # PowerShell passes the pattern through unexpanded; a directory also works
    assert cli.main(["report", str(d / "res" / "*.json"), "--summary"]) == 0
    assert capsys.readouterr().out.count("| city |") == 0  # summary rows are per result
    assert cli.main(["report", str(d / "res"), "--summary"]) == 0
    out = capsys.readouterr().out
    assert (
        sum(line.startswith("| ") for line in out.splitlines()) == 3
    )  # header + 2 result rows


def test_report_with_no_matches_is_a_clean_error(tmp_path, capsys):
    assert cli.main(["report", str(tmp_path / "nothing" / "*.json")]) == 2
    assert "no result files" in capsys.readouterr().err


def test_run_missing_files_give_clean_errors_before_loading_a_model(
    files, monkeypatch, capsys
):
    d, _ = files
    monkeypatch.setattr(
        cli, "make_predictor", lambda args: pytest.fail("model should not load")
    )
    assert (
        cli.main(
            [
                "run",
                f"{d / 'dev.jsonl'},{d / 'test.jsonl'}",
                "--labels",
                "my_labels.yaml",
            ]
        )
        == 2
    )
    assert "labels file not found: my_labels.yaml" in capsys.readouterr().err
    assert (
        cli.main(["run", "dev.jsonl,test.jsonl", "--labels", str(d / "labels.yaml")])
        == 2
    )
    assert "data file not found: dev.jsonl" in capsys.readouterr().err


def test_run_refuses_to_overwrite_without_force(files, monkeypatch, capsys):
    d, gold = files
    monkeypatch.setattr(
        cli, "make_predictor", lambda args: Perfect(gold, {"city": "city"})
    )
    out = d / "r.json"
    out.write_text("{}", encoding="utf-8")
    argv = [
        "run",
        f"{d / 'dev.jsonl'},{d / 'test.jsonl'}",
        "--labels",
        str(d / "labels.yaml"),
        "--out",
        str(out),
    ]
    assert cli.main(argv) == 2
    assert "--force" in capsys.readouterr().err
    assert cli.main([*argv, "--force"]) == 0


def test_bundled_example_runs():
    from pathlib import Path

    ex = Path(__file__).resolve().parents[1] / "examples"
    c = load_labels(str(ex / "labels.yaml"), auto=True)
    validate_candidates(c)
    from labelsweep.data import load_jsonl

    for split in ("dev", "test"):
        rows = load_jsonl(ex / f"{split}.jsonl")
        assert rows and all(t in c for r in rows for *_, t in r.spans)
