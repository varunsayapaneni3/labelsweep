# labelsweep

**How much does your zero-shot NER score depend on how you *word* the entity labels?**

Zero-shot NER models such as [GLiNER](https://github.com/urchade/GLiNER) take the entity types as plain
text (`["person", "organization", "programming language"]`). That makes the wording a hidden
hyperparameter: `"researcher"` vs `"scientist"`, `"organization"` vs `"Organization"`. Related work
shows that adding type descriptions alone changes F1 a lot ([ZeroNER, 2025](https://aclanthology.org/2025.findings-acl.805.pdf)),
and that reported zero-shot scores are inflated by label overlap with training data
([Familiarity, NAACL 2025](https://aclanthology.org/2025.naacl-long.37/)). `labelsweep` measures this for your own
labels and data:

1. For each entity type, it tries every candidate wording while the other types keep their default.
2. It **selects** wordings on a dev split only.
3. It **reports** the spread on a held-out test split, plus the test gain of the chosen label set with a
   paired-bootstrap 95% CI, so you can see whether the "better" wording holds up or just fit the dev set.

## Install

Not on PyPI yet. Install from source:

```bash
git clone https://github.com/<your-username>/labelsweep
cd labelsweep
pip install -e ".[gliner]"        # or: uv sync --extra gliner
```

## Use

```bash
labelsweep presets                                  # bundled benchmarks + label presets
labelsweep run crossner_ai --model urchade/gliner_small-v2.1
labelsweep run examples/dev.jsonl,examples/test.jsonl --labels examples/labels.yaml --auto
labelsweep report results/*.json --summary          # one row per dataset x model (globs work in PowerShell too)
```

`run` will not overwrite an existing result file unless you pass `--force`.
On Windows, if Application Control blocks the `labelsweep.exe` shim, use `python -m labelsweep ...`.

Label file (YAML; first wording = your default). With `--auto`, a type that lists a single wording is
expanded with mechanical variants (lowercase, Title Case, plural, "<x> name"):

```yaml
researcher: [researcher, scientist, academic]
programlang: [programming language, coding language]
person: [person]          # --auto -> person, Person, persons, person name
```

Data file (JSONL, token spans, end-exclusive):

```json
{"tokens": ["Hinton", "works", "at", "Google"], "spans": [[0, 1, "researcher"], [3, 4, "organisation"]]}
```

## Results

Pre-registered study: 6 benchmarks (CrossNER ai, literature, music, politics, science; MIT Restaurant) x
GLiNER v2.1 small / medium / large, full official splits, threshold 0.5. Full tables: [docs/results.md](docs/results.md)
(regenerate with `python scripts/analyze.py`).

| ID | Rule | Observed | Verdict |
|---|---|---|---|
| H1 | median per-type spread >= 5 in all 18 runs | 18/18 runs; range 15.7-47.0 | **supported** |
| H2 | test gain < dev gain and 95% CI includes 0, in >= 10/18 runs | 4/18 (test < dev: 13/18; CI includes 0: 5/18; CI above 0: 12/18; CI below 0: 1/18) | **not supported** |
| H3 | spread small > medium > large on >= 4/6 datasets | 2/6 datasets | **not supported** |
| H4 | casing/plural spread >= 2 in >= 15/18 runs | 18/18 runs; range 6.2-28.0 | **supported** |

What this means in practice:

- **Wording matters a lot.** Rewording a single entity type moves that type's F1 by a median of 15.7-47.0 points
  depending on the run, and capitalisation or a plural alone moves it by 6.2-28.0 points.
- **Choosing wordings on a labelled dev set usually helps, but less than it seems.** The dev-chosen label set improved
  test micro-F1 in 17 of 18 runs (-1.7 to +5.6 points; CI above 0 in 12 runs), and the test gain was smaller than the
  dev gain in 13 of 18. In one run (literature, small) the "better" wording was worse on test.
- **Bigger is not more robust.** Sensitivity fell with model size on only 2 of 6 datasets; on politics it rose
  steadily (28.4 -> 37.9 -> 45.3).

Caveat: a large per-type drop can come from the new wording taking spans away from another type rather than
from missing entities; an error audit of the largest drops is planned ([plan](docs/PROJECT_PLAN.md), task 3.6).

## How it works

- **Data**: CrossNER (5 domains) and MIT Restaurant are read from the Hugging Face parquet export
  (no `datasets` dependency). The official validation split is the dev set and the official test split is the test set.
- **Metric**: strict span match (exact boundaries and type), micro and per-type F1.
- **Search**: a coordinate sweep, where each type's wordings are tried with the other types at default. Predictions are
  cached per label set, so the cost is `1 + sum(candidates - 1) + 1` passes over dev+test.
- **Honesty checks**: per-type support is shown and types with fewer than 30 test spans are flagged; the selected set is
  evaluated jointly on test; the dev gain is printed next to the test gain.

## Limits

- The coordinate sweep ignores interactions between types (the final set is still evaluated jointly).
- The bundled alternatives were written by one person (`scripts/make_presets.py`) and fixed before any results
  ([pre-registration](docs/PREREGISTRATION.md)). Your own alternatives will give different numbers.
- GLiNER backend only (v0.1). Thresholds stay fixed at 0.5.

## Roadmap

GLiNER2 and NuNER backends; LLM-generated paraphrases; type descriptions; does fine-tuning remove wording sensitivity?

## Documentation

- [Project plan](docs/PROJECT_PLAN.md): scope, phases, deliverables, risks, backlog
- [Implementation](docs/IMPLEMENTATION.md): architecture, algorithm, metrics, CLI reference, extending
- [Pre-registration](docs/PREREGISTRATION.md): hypotheses fixed before the full run, with change log

## License

MIT
