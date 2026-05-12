# Artifact Schemas

Every public command should write a machine-readable artifact and a provenance
manifest.

Minimum files:

- `run_manifest.json`: command, git commit, configs, inputs, and outputs.
- `summary.json`: task/model/SAE identifiers, seed, metrics, and checkpoint path.
- `environment.json`: Python version and key dependency versions when practical.

Core record types:

- `native_probe_summary`
- `sae_probe_summary`
- `feature_extraction_summary`
- `sae_code_summary`
- `availability_summary`
- `subset_usage_summary`
- `task_feature_ranking`
- `contribution_scores_summary`
- `feature_ablation_summary`
- `array_contract_validation`
- `artifact_index`
- `reproduction_run_plan`
- `artifact_bundle_check`

The current executable schema contract lives in
`src/feature_economy/artifacts/schemas.py`.

Artifact indexing:

```bash
feature-economy index-artifacts \
  --input-dir /tmp/feature_economy_smoke \
  --output-json /tmp/feature_economy_index/artifact_index.json \
  --output-csv /tmp/feature_economy_index/artifact_index.csv \
  --require-valid
```

The index records each JSON artifact's `record_type`, path, task/model/SAE
identifiers when present, whether it is a smoke artifact, and validation status.
Paper table builders should prefer `artifact_index.json` over ad hoc directory
scans.

Artifact bundle checking:

```bash
feature-economy plan-runs \
  --config-root configs \
  --output-json /tmp/reproduction_run_plan.json \
  --output-csv /tmp/reproduction_run_plan.csv

feature-economy check-bundle \
  --run-plan-json /tmp/reproduction_run_plan.json \
  --artifact-root /path/to/artifact_root \
  --output-json /tmp/artifact_bundle_check.json
```

Use `--require-complete` when a release bundle should fail if any planned row is
missing required files. The checker does not validate metric correctness; it
checks whether each planned artifact directory contains the files required by
that stage, such as `features.npz`, `codes.npz`, `probe_logits.npz`,
`task_feature_ranking.json`, `contribution_scores.npz`, or
`feature_ablation_summary.json`.
