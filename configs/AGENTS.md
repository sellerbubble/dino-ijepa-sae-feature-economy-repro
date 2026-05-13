# Agent Rules For Configs

Configs are the source of truth for the public experiment matrix. Prefer adding
or editing YAML here before adding new code.

## Families

- `models/`: backbone identity, layer aliases, and preprocessing policy.
- `saes/`: SAE identity, model link, layer, TopK/expansion metadata, checkpoint
  placeholder, and activation normalization.
- `tasks/`: dataset/task identity and manifest assumptions.
- `probes/`: paper-scale and smoke probe settings.
- `experiments/`: main paper stages for native probes, SAE probes,
  Availability, Access, and Allocation.
- `sweeps/`: cross-cutting sweeps such as ranking controls and layer sweeps.
- `advanced/`: optional higher-level modules such as Module F native-subspace
  ablation.

## Rules

Use environment-variable placeholders instead of private paths. Keep IDs stable;
artifact directories, reports, and tests depend on them.

After changing configs, run:

```bash
PYTHONPATH=src python -m feature_economy.cli.main check-configs --config-root configs
PYTHONPATH=src python -m feature_economy.cli.main plan-runs --config-root configs --output-json /tmp/feature_economy_plan.json
```

If the run count changes, update `docs/experiment_matrix.md`, saved-array bundle
docs, and tests that assert run-plan row counts.
