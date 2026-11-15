# Pre-registration: label-wording sensitivity of zero-shot NER

Written 2026-11-14, before the full grid was run. Only a 50+50-sentence smoke test on
CrossNER-AI with gliner_small-v2.1 had been seen (used to check the pipeline, not to tune).
Changes after this point are logged at the bottom with date and reason.

## Questions and hypotheses

| ID | Question | Hypothesis | Test |
|---|---|---|---|
| H1 | How much does per-type F1 move when only that type's label wording changes? | Median per-type test spread >= 5 F1 points for every model x dataset | `summarize()` median spread, types with >= 30 test spans |
| H2 | Does picking wordings on a dev set carry over to held-out data? | Test gain < dev gain (selection overfit); 95% CI of the test gain includes 0 on most datasets | paired bootstrap (1000 resamples of test sentences) |
| H3 | Do larger GLiNER models depend less on wording? | Median spread shrinks small > medium > large | compare `median_spread` across the three v2.1 sizes per dataset |
| H4 | Do meaning-preserving edits matter? | Casing or plural alone moves per-type F1 by >= 2 points (median) | `median_trivial_spread` |

## Fixed setup
- Data: CrossNER ai, literature, music, politics, science (MIT licence) and MIT Restaurant; the
  official validation split is the dev set and the official test split is the test set. Full splits.
- Models: urchade/gliner_small-v2.1, gliner_medium-v2.1, gliner_large-v2.1 (Apache-2.0), threshold 0.5, flat NER.
- Candidates: bundled presets from `scripts/make_presets.py` (natural default, 2 hand-written
  alternatives, raw dataset tag, casing variant, plural). Not edited after seeing any result.
- Coordinate sweep: one type reworded at a time, others at default. Selection on dev only.
- Metric: strict span match (exact boundaries and type). Primary: per-type F1 spread on test.
  Secondary: micro-F1 test gain of the dev-chosen label set with paired-bootstrap 95% CI.
- Noise floor: GLiNER inference is deterministic at fixed inputs, so variation comes from the
  wording; sampling uncertainty is covered by the bootstrap and the >= 30-span support filter.

## Known limits (to state in the write-up)
- Coordinate sweep ignores interactions between types (the final label set is evaluated jointly).
- Hand-written alternatives are one author's choices; a different author would write different ones.
- GLiNER v2.1 may have seen similar label strings in training (Golde et al., 2025, "Familiarity").

## Change log
- 2026-11-14: noise floor checked, not assumed. An independent re-run of crossner_ai x gliner_small-v2.1
  (started by hand, GPU shared with the grid) reproduced every score exactly; only wall time differed.
- 2026-11-14: aggregation rules for H2-H4 were not stated above. Added in PROJECT_PLAN.md section 4 after
  16 of 18 runs had completed (deviation, disclosed): H2 supported if test gain < dev gain and the CI
  includes 0 in >= 10 of 18 runs; H3 if small > medium > large on >= 4 of 6 datasets; H4 if casing/plural
  spread >= 2 points in >= 15 of 18 runs. Raw numbers are reported next to each verdict.
