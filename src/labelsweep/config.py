"""Label-candidate files (YAML: gold type -> [default wording, alternatives...])."""

from __future__ import annotations

from importlib import resources

import yaml

from .sweep import auto_variants


def list_presets() -> list[str]:
    return sorted(
        p.name[:-5]
        for p in resources.files("labelsweep.presets").iterdir()
        if p.name.endswith(".yaml")
    )


def load_labels(spec: str, auto: bool = False) -> dict[str, list[str]]:
    """Load candidates from a YAML path or `preset:<name>`. `auto` expands types that list one wording."""
    if spec.startswith("preset:"):
        text = (
            resources.files("labelsweep.presets")
            .joinpath(spec[7:] + ".yaml")
            .read_text(encoding="utf-8")
        )
    else:
        with open(spec, encoding="utf-8") as fh:
            text = fh.read()
    raw = yaml.safe_load(text) or {}
    out: dict[str, list[str]] = {}
    for typ, words in raw.items():
        words = [words] if isinstance(words, str) else [str(w) for w in words]
        out[str(typ)] = auto_variants(words[0]) if auto and len(words) == 1 else words
    return out
