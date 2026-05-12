# Paper Claims Map

This document records how public v0 artifacts map to the paper's main evidence
chain.

| Paper component | Public artifact family | Purpose |
| --- | --- | --- |
| Native task anchor | `native_probe_summary.json` | Compare backbone transfer/readout baselines. |
| SAE analysis substrate | `sae_probe_summary.json` | Establish the sparse feature coordinate system used by later analyses. |
| Availability | `availability_summary.json`, `feature_stats.jsonl` | Measure which SAE features are active and how unevenly they are used. |
| Access | `task_feature_ranking.json`, `subset_usage_summary.json` | Measure which available features tasks recruit. |
| Allocation | `ablation_summary.json`, `ablation_table.csv` | Measure where task performance burden concentrates under feature removal. |
| Paper tables/figures | `paper_tables/*.csv`, `figures/*.pdf` | Reproduce table and overview-figure summaries from saved artifacts. |

Public v0 should keep objective-level explanations conservative. The first
public target is checkpoint-level reproduction of the measured feature economies.

For the canonical command sequence that materializes these artifact families,
see `docs/canonical_chain_runbook.md`.

Current public support:

- `feature-economy probe-native` and `feature-economy probe-sae` can materialize
  fixture summaries or saved-array linear-probe summaries.
- `feature-economy compute-usage`, `rank-features`,
  `compute-subset-usage`, and `ablate-features` can materialize the
  Availability, Access, and Allocation artifacts from saved SAE codes and
  linear-probe outputs.
- Smoke commands still exist as interface tests; they are not scientific
  results.
- `feature-economy index-artifacts` records which artifacts are valid and where
  they live before any paper table or figure consumes them.
- `feature-economy make-tables` can convert indexed public artifacts into
  `probe_scores.csv`, `availability_summary.csv`, `subset_usage_summary.csv`,
  and `ablation_summary.csv`.
- `feature-economy make-figures` can convert those CSV tables into lightweight
  overview figures for probe scores, availability, access, and allocation.
- Full paper table reproduction still requires real saved arrays, checkpoints,
  and manifests rather than the tiny fixture path.
