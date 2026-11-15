"""Examples, BIO decoding and dataset loaders.

Spans are token-level, end-exclusive: (start, end, type).
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

Span = tuple[int, int, str]


@dataclass
class Example:
    tokens: list[str]
    spans: list[Span] = field(default_factory=list)


def bio_to_spans(tags: list[str]) -> list[Span]:
    """Decode BIO/IOB2 tags into spans. A stray I- tag starts a new span."""
    spans: list[Span] = []
    start, cur = None, None
    for i, tag in enumerate(tags):
        prefix, _, typ = tag.partition("-")
        if tag == "O" or prefix not in ("B", "I"):
            if cur is not None:
                spans.append((start, i, cur))
            start, cur = None, None
            continue
        if prefix == "B" or cur != typ:
            if cur is not None:
                spans.append((start, i, cur))
            start, cur = i, typ
    if cur is not None:
        spans.append((start, len(tags), cur))
    return spans


def load_jsonl(path: str | Path) -> list[Example]:
    """One JSON object per line: {"tokens": [...], "spans": [[start, end, "type"], ...]}."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                out.append(
                    Example(list(d["tokens"]), [tuple(s) for s in d.get("spans", [])])
                )
    return out


# --- Hugging Face (parquet export via the datasets-server API; no `datasets` dependency) ---

BENCHMARKS: dict[str, tuple[str, str]] = {
    # name: (hf dataset id, config)
    "crossner_ai": ("DFKI-SLT/cross_ner", "ai"),
    "crossner_literature": ("DFKI-SLT/cross_ner", "literature"),
    "crossner_music": ("DFKI-SLT/cross_ner", "music"),
    "crossner_politics": ("DFKI-SLT/cross_ner", "politics"),
    "crossner_science": ("DFKI-SLT/cross_ner", "science"),
    "mit_restaurant": ("tner/mit_restaurant", "mit_restaurant"),
    "mit_movie": ("tner/mit_movie_trivia", "mit_movie_trivia"),
}

_UA = {"User-Agent": "labelsweep"}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=_UA), timeout=120
    ) as r:
        return r.read()


def _cache_dir() -> Path:
    import os

    d = Path(os.environ.get("LABELSWEEP_CACHE", Path.home() / ".cache" / "labelsweep"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _tag_names(dataset: str, config: str) -> list[str]:
    info = json.loads(
        _get(
            f"https://datasets-server.huggingface.co/info?dataset={dataset}&config={config}"
        )
    )
    feats = info["dataset_info"]["features"]
    tags = feats["ner_tags"]
    inner = tags.get("feature") or tags.get("sequence") or tags
    if "names" in inner:
        return list(inner["names"])
    # tner datasets store ints without class names; fall back to their label.json
    label2id = json.loads(
        _get(f"https://huggingface.co/datasets/{dataset}/raw/main/dataset/label.json")
    )
    return [k for k, _ in sorted(label2id.items(), key=lambda kv: kv[1])]


def load_hf(
    name_or_id: str, split: str = "test", config: str | None = None
) -> list[Example]:
    """Load a BIO-tagged token-classification split from the Hugging Face parquet export."""
    import pyarrow.parquet as pq

    dataset, cfg = BENCHMARKS.get(name_or_id, (name_or_id, config or "default"))
    if config:
        cfg = config
    names = _tag_names(dataset, cfg)
    urls = json.loads(
        _get(f"https://huggingface.co/api/datasets/{dataset}/parquet/{cfg}/{split}")
    )
    examples: list[Example] = []
    for k, url in enumerate(urls):
        local = (
            _cache_dir() / f"{dataset.replace('/', '__')}__{cfg}__{split}__{k}.parquet"
        )
        if not local.exists():
            local.write_bytes(_get(url))
        table = pq.read_table(local, columns=["tokens", "ner_tags"]).to_pylist()
        for row in table:
            tags = [names[t] for t in row["ner_tags"]]
            examples.append(Example(list(row["tokens"]), bio_to_spans(tags)))
    return examples


def entity_types(examples: list[Example]) -> list[str]:
    """Types that occur in the data, most frequent first."""
    from collections import Counter

    c = Counter(t for ex in examples for _, _, t in ex.spans)
    return [t for t, _ in c.most_common()]
