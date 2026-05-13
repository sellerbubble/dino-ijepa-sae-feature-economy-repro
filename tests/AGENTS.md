# Agent Rules For Tests

Tests must remain small, deterministic, and public-data-free. They validate
contracts, command wiring, schemas, and tiny numerical behavior rather than
paper-scale performance.

## Rules

Use temporary directories and tiny arrays. Do not require datasets, GPUs,
network access, model checkpoints, SAE checkpoints, or private paths.

When adding a new command, add at least one test that verifies:

- command/function output exists;
- summary JSON validates through `artifacts/schemas.py`;
- `run_manifest.json` is emitted when applicable;
- failure modes are explicit for missing required inputs.

Run:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```
